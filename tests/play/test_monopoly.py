"""Mechanics tests for the plain-game MonopolyLite (play/games/monopoly_lite.py).

This is the Task-3 rebuild: no negotiation phase, no alliances, dice are
auto-rolled inside ``_begin_turn`` (there is no ``roll`` action), a buy
decision follows an unowned-landing, and an end-of-turn trade dialogue
(propose -> respond) closes each turn.

Task 4 adds the text-parsing grammar / prompts; this file covers the core
state machine and mechanics only.  Trade actions are exercised by driving
``step`` directly (RandomPlayer only ever sees ``no_trade`` in
``PH_TRADE_PROPOSE`` until Task 4 wires up proposal parsing).
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from game_theory_llm.play.games.monopoly_lite import (
    MonopolyLite,
    MLState,
    BOARD,
    BOARD_SIZE,
    GROUPS,
    PROP_INFO,
    STARTING_CASH,
    GO_SALARY,
    PH_BUY,
    PH_TRADE_PROPOSE,
    PH_TRADE_RESPOND,
    PH_TERMINAL,
)
from game_theory_llm.play.base import Obs, ParseError
from game_theory_llm.play.players import RandomPlayer
from game_theory_llm.play import run_match


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

PLAYERS = ["Alice", "Bob", "Carol", "Dave", "Eve"]


def make_game(players=None, seed: int = 42, turn_cap: int = 200) -> MonopolyLite:
    return MonopolyLite(players=players or PLAYERS, seed=seed, turn_cap=turn_cap)


def _initial_state(game: MonopolyLite, seed: int = 0) -> MLState:
    return game.initial_state(random.Random(seed))


def _fresh_pre_turn_state(game: MonopolyLite, seed: int = 0) -> MLState:
    """Build an initial state, then reset it to a clean pre-decision baseline
    for hand-crafted mechanics tests: everyone on GO, full cash, all props
    unowned, PH_TRADE_PROPOSE for seat 0, no auto-roll side effects."""
    state = _initial_state(game, seed)
    for name in state.player_names:
        state.positions[name] = 0
        state.cash[name] = STARTING_CASH
    for pid in state.properties:
        state.properties[pid] = None
    state.bankrupt = set()
    state.current_player_idx = 0
    state.phase = PH_TRADE_PROPOSE
    state.pending_buy_square = None
    state.pending_trade = None
    state.traded_this_turn = False
    state.winner = None
    state.turn = 0
    state.events = []
    state.new_events = []
    return state


def _force_landing(game: MonopolyLite, state: MLState, target_sq: int) -> int:
    """Set the current player's position so the deterministic roll for the
    current (turn, seat) lands them exactly on ``target_sq``.  ``target_sq``
    must be >= 12 to guarantee no GO wrap (max 2d6 roll is 12)."""
    assert target_sq >= 12, "pick a target square >= 12 to avoid GO-wrap"
    idx = state.current_player_idx
    name = state.player_names[idx]
    roll = game._roll_dice(state)
    state.positions[name] = (target_sq - roll) % BOARD_SIZE
    return roll


def _run(seed: int, *, turn_cap: int = 30, max_turns: int = 5000):
    """Run a full RandomPlayer match; return (game, result, events)."""
    game = make_game(seed=seed, turn_cap=turn_cap)
    players = [RandomPlayer(seed=seed * 31 + i) for i in range(game.n_players)]
    tmp = tempfile.mkdtemp()
    log_path = Path(tmp) / f"monopoly_{seed}.jsonl"
    result = run_match(game, players, seed=seed, log_path=log_path,
                       max_turns=max_turns)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    return game, result, events


# --------------------------------------------------------------------------- #
# Board layout (static)
# --------------------------------------------------------------------------- #

class TestBoardLayout:
    def test_board_has_20_squares(self):
        assert len(BOARD) == 20
        assert BOARD_SIZE == 20

    def test_all_squares_have_sq_index(self):
        for i, sq in enumerate(BOARD):
            assert sq["sq"] == i

    def test_all_prop_ids_unique(self):
        ids = [sq["prop_id"] for sq in BOARD if "prop_id" in sq]
        assert len(ids) == len(set(ids))

    def test_all_groups_correct_size(self):
        for group, pids in GROUPS.items():
            assert len(pids) == (4 if group == "RAILROAD" else 2)

    def test_starting_constants(self):
        assert STARTING_CASH == 1500
        assert GO_SALARY == 200


# --------------------------------------------------------------------------- #
# Auto-roll: the defining property of the plain-game rebuild
# --------------------------------------------------------------------------- #

class TestAutoRoll:
    def test_initial_state_auto_rolls_into_a_decision_phase(self):
        """initial_state must run the first _begin_turn: a die was rolled and
        the FIRST active_player is a decision phase (not a roll prompt)."""
        game = make_game()
        state = _initial_state(game)
        assert state.dice_roll is not None, "no die was auto-rolled"
        assert state.phase in (PH_BUY, PH_TRADE_PROPOSE), state.phase
        assert game.active_player(state) == 0, "seat 0 must own the first decision"

    def test_no_roll_action_anywhere(self):
        """A `roll` action must no longer exist in any legal-action set."""
        game = make_game(seed=99, turn_cap=40)
        state = _initial_state(game, seed=99)
        rng = random.Random(0)
        for _ in range(300):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            assert active >= 0
            legal = game.legal_actions(state, active)
            assert legal
            assert all(a.get("type") != "roll" for a in legal), legal
            state = game.step(state, rng.choice(legal))

    def test_current_player_moved_others_untouched(self):
        """After the auto-roll, seat 0 has moved to its rolled square; the
        other seats are still on GO with full cash."""
        game = make_game()
        state = _initial_state(game)
        mover = state.player_names[0]
        # roll <= 12 < 20, so no GO wrap for a player starting on 0.
        assert state.positions[mover] == state.dice_roll
        for name in state.player_names[1:]:
            assert state.positions[name] == 0
            assert state.cash[name] == STARTING_CASH

    def test_legal_action_shapes(self):
        """legal_actions must return the Task-4 fixed shapes per phase."""
        game = make_game(players=["Alice", "Bob"])
        state = _fresh_pre_turn_state(game)

        # PH_TRADE_PROPOSE
        state.phase = PH_TRADE_PROPOSE
        assert game.legal_actions(state, 0) == [{"type": "no_trade"}]

        # PH_BUY (must include both, unaffordability is filtered upstream)
        state.phase = PH_BUY
        state.pending_buy_square = 16
        assert game.legal_actions(state, 0) == [
            {"type": "decline"}, {"type": "buy"}]

        # PH_TRADE_RESPOND
        state.phase = PH_TRADE_RESPOND
        state.pending_trade = {
            "from": "Alice", "to": "Bob", "give_props": [], "give_cash": 0,
            "want_props": [], "want_cash": 0, "message": "",
        }
        bob = 1
        assert game.legal_actions(state, bob) == [
            {"type": "accept_trade"}, {"type": "reject_trade"}]

    def test_no_mixins_no_nego_no_alli(self):
        """The rebuilt state must not carry negotiation / alliance sub-state."""
        game = make_game()
        state = _initial_state(game)
        assert not hasattr(state, "nego")
        assert not hasattr(state, "alli")
        # Class must not inherit the messaging / alliance mixins.
        mro_names = {c.__name__ for c in type(game).__mro__}
        assert "MessagingMixin" not in mro_names
        assert "AllianceMixin" not in mro_names


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

class TestDeterminism:
    def test_same_seed_same_initial_state(self):
        g1 = MonopolyLite(players=PLAYERS, seed=7)
        g2 = MonopolyLite(players=PLAYERS, seed=7)
        st1 = g1.initial_state(random.Random(7))
        st2 = g2.initial_state(random.Random(7))
        assert st1.player_names == st2.player_names
        assert st1.cash == st2.cash
        assert st1.positions == st2.positions
        assert st1.properties == st2.properties
        assert st1.phase == st2.phase
        assert st1.dice_roll == st2.dice_roll
        assert st1.events == st2.events

    def test_different_seeds_differ(self):
        dice_by_seed = {}
        for s in range(10):
            game = MonopolyLite(players=PLAYERS, seed=s, turn_cap=5)
            state = game.initial_state(random.Random(s))
            rolls = []
            for turn in range(4):
                state.turn = turn
                rolls.append(game._roll_dice(state))
            dice_by_seed[s] = tuple(rolls)
        assert len(set(dice_by_seed.values())) > 1


# --------------------------------------------------------------------------- #
# GO salary
# --------------------------------------------------------------------------- #

class TestGOSalary:
    def test_pass_go_awards_salary(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        alice = state.player_names[0]
        state.positions[alice] = 18  # 18 + any roll (>=2) wraps past GO
        state.turn = 5
        cash_before = state.cash[alice]
        game._begin_turn(state)
        assert any("GO, collected" in e for e in state.events), state.events
        # Cash reflects the salary (possibly minus a tax/rent on the landing).
        assert state.cash[alice] >= cash_before + GO_SALARY - 100


# --------------------------------------------------------------------------- #
# Tax and rent
# --------------------------------------------------------------------------- #

class TestTaxAndRent:
    def test_pay_to_bank_disappears(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        payer = "Bob"
        before = state.cash[payer]
        game._pay(state, payer, None, 100)
        assert state.cash[payer] == before - 100
        # No creditor was credited (bank payment).
        assert sum(state.cash.values()) == (
            STARTING_CASH * len(state.player_names) - 100)

    def test_tax_square_landing_charges_bank(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        alice = state.player_names[0]
        before = state.cash[alice]
        _force_landing(game, state, 14)  # sq 14 = tax $75
        game._begin_turn(state)
        assert state.positions[alice] == 14
        assert state.cash[alice] == before - 75
        assert state.phase == PH_TRADE_PROPOSE

    def test_single_ownership_base_rent(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        state.properties["purple1"] = "Alice"
        sq = PROP_INFO["purple1"]
        assert game._compute_rent(state, sq, "Alice") == sq["rent"]

    def test_full_group_doubles_rent(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        for group_name, pids in GROUPS.items():
            if group_name == "RAILROAD":
                continue
            for pid in pids:
                state.properties[pid] = "Alice"
            sq = PROP_INFO[pids[0]]
            assert game._compute_rent(state, sq, "Alice") == sq["full_rent"]
            for pid in pids:
                state.properties[pid] = None

    def test_railroad_full_set_doubles_rent(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        for pid in GROUPS["RAILROAD"]:
            state.properties[pid] = "Alice"
        sq = PROP_INFO["rr1"]
        assert game._compute_rent(state, sq, "Alice") == sq["full_rent"]

    def test_partial_group_no_doubling(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        sq = PROP_INFO["purple1"]
        assert game._compute_rent(state, sq, "Alice") == sq["rent"]

    def test_rent_paid_to_owner_on_landing(self):
        """Landing on an opponent's monopoly charges the doubled rent to the
        owner."""
        game = make_game()
        state = _fresh_pre_turn_state(game)
        # Alice owns the whole RED group (sq 16 red1, sq 18 red2) -> full_rent.
        state.properties["red1"] = "Alice"
        state.properties["red2"] = "Alice"
        # Bob is the current roller.
        state.current_player_idx = 1
        bob = "Bob"
        alice_before = state.cash["Alice"]
        bob_before = state.cash[bob]
        _force_landing(game, state, 16)  # red1
        game._begin_turn(state)
        full = PROP_INFO["red1"]["full_rent"]
        assert state.cash[bob] == bob_before - full
        assert state.cash["Alice"] == alice_before + full


# --------------------------------------------------------------------------- #
# Buy decision
# --------------------------------------------------------------------------- #

class TestBuyDecision:
    def test_unowned_affordable_opens_buy_phase(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        _force_landing(game, state, 16)  # red1, unowned, price 200
        game._begin_turn(state)
        assert state.phase == PH_BUY
        assert state.pending_buy_square == 16
        assert game.active_player(state) == 0

    def test_buy_transfers_cash_and_deed_then_trade_phase(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        _force_landing(game, state, 16)
        game._begin_turn(state)
        alice = state.player_names[0]
        before = state.cash[alice]
        price = PROP_INFO["red1"]["price"]
        game.step(state, {"type": "buy"})
        assert state.properties["red1"] == alice
        assert state.cash[alice] == before - price
        assert state.phase == PH_TRADE_PROPOSE          # same player, NOT advance
        assert state.current_player_idx == 0
        assert any(f"bought red1 for ${price}" in e for e in state.events)

    def test_decline_leaves_state_and_moves_to_trade_phase(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        _force_landing(game, state, 16)
        game._begin_turn(state)
        alice = state.player_names[0]
        before = state.cash[alice]
        game.step(state, {"type": "decline"})
        assert state.properties["red1"] is None
        assert state.cash[alice] == before
        assert state.phase == PH_TRADE_PROPOSE
        assert state.current_player_idx == 0
        assert any("declined to buy red1" in e for e in state.events)

    def test_unaffordable_auto_skips_with_event(self):
        """If the roller cannot afford the unowned square, no PH_BUY is
        entered: an event is logged and the turn falls through to trade."""
        game = make_game()
        state = _fresh_pre_turn_state(game)
        alice = state.player_names[0]
        state.cash[alice] = 50  # red1 costs 200
        _force_landing(game, state, 16)
        game._begin_turn(state)
        assert state.phase == PH_TRADE_PROPOSE
        assert state.pending_buy_square is None
        assert any("cannot afford red1 ($200)" in e for e in state.events)


# --------------------------------------------------------------------------- #
# Trade lifecycle (driven through step directly)
# --------------------------------------------------------------------------- #

class TestTradeLifecycle:
    def _propose_state(self, turn: int = 0):
        game = make_game(players=["Alice", "Bob", "Carol"], turn_cap=200)
        state = _fresh_pre_turn_state(game)
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.cash["Alice"] = 800
        state.cash["Bob"] = 600
        state.phase = PH_TRADE_PROPOSE
        state.current_player_idx = 0
        state.turn = turn
        return game, state

    def test_propose_stores_pending_and_switches_to_respond(self):
        game, state = self._propose_state()
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 100,
            "want_props": ["purple2"], "want_cash": 0,
            "message": "let's deal",
        })
        assert state.phase == PH_TRADE_RESPOND
        assert state.traded_this_turn is True
        pt = state.pending_trade
        assert pt["from"] == "Alice" and pt["to"] == "Bob"
        assert pt["give_props"] == ["purple1"] and pt["give_cash"] == 100
        assert pt["want_props"] == ["purple2"]
        assert pt["message"] == "let's deal"
        # Recipient is the active player during PH_TRADE_RESPOND.
        assert game.active_player(state) == state.player_names.index("Bob")
        # Event lines: a proposal line + the spoken message line.
        assert any(e.startswith("Alice proposed trade to Bob:") for e in state.events)
        assert 'Alice says: "let\'s deal"' in state.events

    def test_accept_executes_exact_transfers(self):
        """Accept at turn_cap-1 so the follow-on advance goes terminal and the
        post-trade balances are not perturbed by the next player's roll."""
        game, state = self._propose_state(turn=199)  # turn_cap default 200
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 100,
            "want_props": ["purple2"], "want_cash": 0, "message": "",
        })
        game.step(state, {"type": "accept_trade", "message": "ok"})
        assert state.properties["purple1"] == "Bob"
        assert state.properties["purple2"] == "Alice"
        assert state.cash["Alice"] == 800 - 100
        assert state.cash["Bob"] == 600 + 100
        assert state.pending_trade is None
        assert any(e.startswith("Trade completed:") for e in state.events)
        assert 'Bob says: "ok"' in state.events

    def test_accept_two_way_cash_exact_no_clamping(self):
        game, state = self._propose_state(turn=199)
        state.cash["Alice"] = 500
        state.cash["Bob"] = 400
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": [], "give_cash": 200,
            "want_props": [], "want_cash": 100, "message": "",
        })
        game.step(state, {"type": "accept_trade", "message": ""})
        assert state.cash["Alice"] == 500 - 200 + 100
        assert state.cash["Bob"] == 400 + 200 - 100

    def test_reject_leaves_state_unchanged(self):
        game, state = self._propose_state(turn=199)
        props0 = dict(state.properties)
        a0, b0 = state.cash["Alice"], state.cash["Bob"]
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 50,
            "want_props": ["purple2"], "want_cash": 0, "message": "",
        })
        game.step(state, {"type": "reject_trade", "message": "no"})
        assert state.properties == props0
        assert state.cash["Alice"] == a0
        assert state.cash["Bob"] == b0
        assert state.pending_trade is None
        assert "Bob rejected the trade" in state.events

    def test_accept_insufficient_funds_becomes_rejection(self):
        game, state = self._propose_state(turn=199)
        state.cash["Bob"] = 10  # cannot cover want_cash of 100
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 0,
            "want_props": [], "want_cash": 100, "message": "",
        })
        game.step(state, {"type": "accept_trade", "message": ""})
        # No transfer occurred; purple1 still Alice's.
        assert state.properties["purple1"] == "Alice"
        assert state.cash["Bob"] == 10
        assert state.pending_trade is None
        assert "trade failed: insufficient funds" in state.events

    def test_no_trade_advances_turn(self):
        game, state = self._propose_state(turn=0)
        game.step(state, {"type": "no_trade"})
        # Advanced off seat 0 to the next solvent seat (and began their turn).
        assert state.current_player_idx != 0
        assert state.traded_this_turn is False
        assert state.turn == 1

    def test_second_proposal_same_turn_rejected_by_phase_assert(self):
        """After a proposal the phase is PH_TRADE_RESPOND; a second
        propose_trade in the same turn is not a legal phase transition and
        step must assert."""
        game, state = self._propose_state()
        game.step(state, {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 0,
            "want_props": ["purple2"], "want_cash": 0, "message": "",
        })
        with pytest.raises(AssertionError):
            game.step(state, {
                "type": "propose_trade", "to": "Bob",
                "give_props": ["purple1"], "give_cash": 0,
                "want_props": ["purple2"], "want_cash": 0, "message": "",
            })


# --------------------------------------------------------------------------- #
# Bankruptcy
# --------------------------------------------------------------------------- #

class TestBankruptcy:
    def test_rent_bankruptcy_transfers_assets_to_creditor(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        state.properties["red1"] = "Alice"
        state.properties["red2"] = "Alice"  # full group -> rent 50
        state.cash["Bob"] = 30
        state.properties["purple1"] = "Bob"
        alice0 = state.cash["Alice"]
        rent = game._compute_rent(state, PROP_INFO["red1"], "Alice")
        game._pay(state, "Bob", "Alice", rent)
        assert "Bob" in state.bankrupt
        assert state.cash["Bob"] == 0
        assert state.cash["Alice"] == alice0 + 30
        assert state.properties["purple1"] == "Alice"

    def test_tax_bankruptcy_sends_assets_to_bank(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        state.cash["Carol"] = 10
        state.properties["orange1"] = "Carol"
        game._pay(state, "Carol", None, 100)
        assert "Carol" in state.bankrupt
        assert state.cash["Carol"] == 0
        assert state.properties["orange1"] is None

    def test_last_solvent_player_wins_immediately(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        for name in ["Bob", "Carol", "Dave"]:
            state.bankrupt.add(name)
        state.cash["Eve"] = 5
        game._bankrupt(state, "Eve", None)
        assert state.winner == "Alice"
        assert game.is_terminal(state)

    def test_bankrupt_player_never_active_in_playout(self):
        game = make_game(seed=3, turn_cap=25)
        state = _initial_state(game, seed=3)
        rng = random.Random(42)
        for _ in range(600):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            assert active >= 0
            assert state.player_names[active] not in state.bankrupt
            legal = game.legal_actions(state, active)
            assert legal
            state = game.step(state, rng.choice(legal))
            for name in state.player_names:
                if name not in state.bankrupt:
                    assert state.cash.get(name, 0) >= 0
            for pid, owner in state.properties.items():
                if owner is not None:
                    assert owner in state.player_names


# --------------------------------------------------------------------------- #
# Turn cap + winner tie-break
# --------------------------------------------------------------------------- #

class TestTurnCap:
    def test_turn_cap_richest_net_worth_wins(self):
        game = make_game(players=["Alice", "Bob", "Carol"], turn_cap=5)
        state = _fresh_pre_turn_state(game)
        state.cash = {"Alice": 100, "Bob": 300, "Carol": 50}
        state.turn = 4  # advance -> turn 5 == cap
        game._advance_turn(state)
        assert state.phase == PH_TERMINAL
        assert state.winner == "Bob"

    def test_tie_break_higher_cash_beats_lower_seat(self):
        """Equal net worth: higher CASH wins even against a lower seat index."""
        game = make_game(players=["Alice", "Bob", "Carol"], turn_cap=5)
        state = _fresh_pre_turn_state(game)
        # Alice (seat 0): cash 40 + purple1 ($60) -> net worth 100.
        # Bob   (seat 1): cash 100                -> net worth 100.
        state.cash = {"Alice": 40, "Bob": 100, "Carol": 50}
        state.properties["purple1"] = "Alice"
        assert game._net_worth(state, "Alice") == game._net_worth(state, "Bob")
        state.turn = 4
        game._advance_turn(state)
        assert state.winner == "Bob", "higher cash must beat lower seat index"

    def test_tie_break_lower_seat_when_cash_equal(self):
        game = make_game(players=["Alice", "Bob", "Carol"], turn_cap=5)
        state = _fresh_pre_turn_state(game)
        state.cash = {"Alice": 100, "Bob": 100, "Carol": 50}
        state.turn = 4
        game._advance_turn(state)
        assert state.winner == "Alice", "equal net worth & cash -> lower seat"


# --------------------------------------------------------------------------- #
# Playout invariants / termination
# --------------------------------------------------------------------------- #

class TestPlayoutInvariants:
    def test_200_step_random_playout_all_actions_legal(self):
        game = make_game(seed=99, turn_cap=50)
        state = _initial_state(game, seed=99)
        rng = random.Random(0)
        for step_no in range(200):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            assert active >= 0, f"step {step_no}: active=-1 in non-terminal"
            legal = game.legal_actions(state, active)
            assert legal, f"step {step_no}: empty legal_actions"
            action = rng.choice(legal)
            assert action in legal
            state = game.step(state, action)
            for name in state.player_names:
                if name not in state.bankrupt:
                    assert state.cash.get(name, 0) >= 0
            for pid, owner in state.properties.items():
                if owner is not None:
                    assert owner in state.player_names

    def test_legal_actions_empty_for_non_active(self):
        game = make_game()
        state = _initial_state(game)
        active = game.active_player(state)
        for seat in range(game.n_players):
            if seat != active:
                assert game.legal_actions(state, seat) == []

    def test_active_player_never_negative_until_terminal(self):
        """The runner's advance_phase branch must never trigger: every
        non-terminal state has a decision owner."""
        game = make_game(seed=5, turn_cap=40)
        state = _initial_state(game, seed=5)
        rng = random.Random(11)
        for _ in range(500):
            if game.is_terminal(state):
                break
            assert game.active_player(state) >= 0
            active = game.active_player(state)
            legal = game.legal_actions(state, active)
            state = game.step(state, rng.choice(legal))

    @pytest.mark.parametrize("seed", range(5))
    def test_random_match_terminates_with_winner(self, seed):
        game, result, events = _run(seed, turn_cap=30, max_turns=4000)
        st = result.terminal_state
        assert game.is_terminal(st)
        assert result.n_turns < 4000
        assert st.winner in st.player_names
        rewards = game.rewards(st)
        assert len(rewards) == game.n_players
        assert sum(rewards) == 1.0
        assert rewards[st.player_names.index(st.winner)] == 1.0


# --------------------------------------------------------------------------- #
# Observations / event broadcasting
# --------------------------------------------------------------------------- #

class TestObservations:
    def test_new_events_drained_once_to_all_seats(self):
        game = make_game()
        state = _initial_state(game)
        active = game.active_player(state)
        action = game.legal_actions(state, active)[0]
        state = game.step(state, action)
        n_before = len(state.new_events)
        assert n_before > 0, "expected accumulated events (auto-roll + step)"
        obs = game.observations(state, state, action, active)
        event_obs = [o for o in obs if o.payload["type"] == "event"]
        assert len(event_obs) == n_before
        for o in event_obs:
            assert set(o.audience) == set(range(game.n_players))
            assert "text" in o.payload
        # Drained (cleared) -> a re-drain yields no further event obs.
        assert state.new_events == []
        obs2 = game.observations(state, state, action, active)
        assert [o for o in obs2 if o.payload["type"] == "event"] == []

    def test_event_lines_recorded_in_permanent_events(self):
        game = make_game()
        state = _fresh_pre_turn_state(game)
        _force_landing(game, state, 16)
        game._begin_turn(state)
        game.step(state, {"type": "buy"})
        # Permanent log accumulates; new_events is the per-batch delta.
        assert any("rolled" in e for e in state.events)
        assert any("bought red1" in e for e in state.events)

    def test_trade_dialogue_obs_on_trade_steps(self):
        game = make_game(players=["Alice", "Bob", "Carol"])
        state = _fresh_pre_turn_state(game)
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.phase = PH_TRADE_PROPOSE
        state.current_player_idx = 0
        action = {
            "type": "propose_trade", "to": "Bob",
            "give_props": ["purple1"], "give_cash": 0,
            "want_props": ["purple2"], "want_cash": 0, "message": "deal?",
        }
        game.step(state, action)
        obs = game.observations(state, state, action, 0)
        td = [o for o in obs if o.payload["type"] == "trade_dialogue"]
        assert len(td) == 1
        payload = td[0].payload
        assert set(td[0].audience) == set(range(game.n_players))
        assert payload["event"] == "propose_trade"
        assert payload["trade"]["to"] == "Bob"
        assert payload["message"] == "deal?"

    def test_run_match_broadcasts_events(self):
        """End-to-end: the runner logs observation records carrying event
        payloads for the auto-rolled / resolved turns."""
        game, result, events = _run(1, turn_cap=15, max_turns=3000)
        obs_records = [e for e in events if e.get("type") == "observation"]
        assert obs_records, "no observation records were logged"
        event_payloads = [
            e for e in obs_records if e.get("obs", {}).get("type") == "event"
        ]
        assert event_payloads, "no event observations were broadcast"


# --------------------------------------------------------------------------- #
# Watcher hooks / integration
# --------------------------------------------------------------------------- #

class TestWatcherHooks:
    def test_name_and_player_count(self):
        game = make_game()
        assert game.name == "monopoly_lite"
        g3 = MonopolyLite(players=["X", "Y", "Z"], seed=0)
        assert g3.n_players == 3
        assert len(g3.initial_state(random.Random(0)).player_names) == 3

    def test_god_view_includes_events_tail(self):
        game = make_game()
        state = _initial_state(game)
        gv = game.god_view(state)
        assert "cash" in gv and "properties" in gv
        assert "events_tail" in gv
        assert isinstance(gv["events_tail"], list)
        assert len(gv["events_tail"]) <= 20

    def test_snapshot_shape(self):
        game = make_game()
        state = _initial_state(game)
        snap = game.snapshot(state)
        assert "public" in snap and "hidden" in snap
        assert "positions" in snap["public"]
        assert "cash" in snap["hidden"]

    def test_render_board_contains_title(self):
        game = make_game()
        state = _initial_state(game)
        board = game.render_board(state)
        assert "Monopoly Lite" in board

    def test_render_prompt_returns_string(self):
        game = make_game()
        state = _initial_state(game)
        for seat in range(game.n_players):
            prompt = game.render_prompt(state, seat)
            assert isinstance(prompt, str) and len(prompt) > 10

    def test_parse_action_is_task4_placeholder(self):
        game = make_game()
        state = _initial_state(game)
        with pytest.raises(ParseError):
            game.parse_action(state, 0, "buy")
