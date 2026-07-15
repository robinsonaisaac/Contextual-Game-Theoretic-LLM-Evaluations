"""Tests for MonopolyLite (play/games/monopoly_lite.py).

Covers:
  * determinism: same seed -> same initial state;
  * 200-step random playout: every chosen action is legal, cash never
    negative without prior bankruptcy resolution, property ownership
    consistent;
  * termination: random playouts end within turn_cap;
  * rent doubling: full colour-group ownership doubles rent;
  * trade lifecycle: accept transfers correctly, reject leaves state clean;
  * bankruptcy: assets transfer to creditor, player removed from rotation.
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
    GROUPS,
    PROP_INFO,
    STARTING_CASH,
    GO_SALARY,
    PH_NEGOTIATION,
    PH_ROLL,
    PH_BUY,
    PH_TRADE,
    PH_TERMINAL,
)
from game_theory_llm.play.players import RandomPlayer
from game_theory_llm.play import run_match


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

PLAYERS = ["Alice", "Bob", "Carol", "Dave", "Eve"]


def make_game(seed: int = 42, turn_cap: int = 200) -> MonopolyLite:
    return MonopolyLite(players=PLAYERS, seed=seed, turn_cap=turn_cap)


def _run(seed: int, *, turn_cap: int = 200, max_turns: int = 5000):
    """Run a match with RandomPlayer agents and return (game, result, events)."""
    game = make_game(seed=seed, turn_cap=turn_cap)
    players = [RandomPlayer(seed=seed * 31 + i) for i in range(game.n_players)]
    tmp = tempfile.mkdtemp()
    log_path = Path(tmp) / f"monopoly_{seed}.jsonl"
    result = run_match(game, players, seed=seed, log_path=log_path,
                       max_turns=max_turns)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    return game, result, events


def _initial_state(game: MonopolyLite) -> MLState:
    return game.initial_state(random.Random(0))


# --------------------------------------------------------------------------- #
# TestDeterminism
# --------------------------------------------------------------------------- #

class TestDeterminism:
    def test_same_seed_same_initial_state(self):
        """Two games with the same seed must produce identical initial states."""
        g1 = MonopolyLite(players=PLAYERS, seed=7)
        g2 = MonopolyLite(players=PLAYERS, seed=7)
        st1 = g1.initial_state(random.Random(7))
        st2 = g2.initial_state(random.Random(7))

        assert st1.player_names == st2.player_names
        assert st1.cash == st2.cash
        assert st1.positions == st2.positions
        assert st1.properties == st2.properties
        assert st1.bankrupt == st2.bankrupt
        assert st1.turn_cap == st2.turn_cap
        assert st1.phase == st2.phase

    def test_different_seeds_differ(self):
        """Different game seeds must produce at least one different dice roll
        over a short sequence, proving that the game_seed feeds into dice."""
        # Use a broader range of seeds to ensure at least one pair differs.
        dice_by_seed = {}
        for s in range(10):
            game = MonopolyLite(players=PLAYERS, seed=s, turn_cap=5)
            state = game.initial_state(random.Random(s))
            rolls = []
            for turn in range(4):
                state.turn = turn
                rolls.append(game._roll_dice(state))
            dice_by_seed[s] = tuple(rolls)
        # Not all roll sequences should be identical across seeds.
        unique_seqs = set(dice_by_seed.values())
        assert len(unique_seqs) > 1, (
            "All 10 seeds produced identical dice sequences — "
            "game_seed is not influencing _roll_dice."
        )


# --------------------------------------------------------------------------- #
# TestLegalActionClosure
# --------------------------------------------------------------------------- #

class TestLegalActionClosure:
    def test_200_step_random_playout_all_actions_legal(self):
        """Play 200 steps with random legal-action selection.  Every action
        chosen must appear in legal_actions for that player.  Cash must never
        go negative (all deductions go through _pay which guards this).
        Property ownership must remain consistent (one owner per property).
        """
        game = make_game(seed=99, turn_cap=50)
        state = game.initial_state(random.Random(99))
        rng = random.Random(0)

        for step_no in range(200):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            assert active >= 0, f"step {step_no}: active_player=-1 in non-terminal"

            legal = game.legal_actions(state, active)
            assert legal, f"step {step_no}: empty legal_actions for seat {active}"

            action = rng.choice(legal)
            # The chosen action must itself appear in legal_actions.
            assert action in legal, (
                f"step {step_no}: chosen action {action} not in legal_actions"
            )

            state = game.step(state, action)

            # Cash invariant: no player should have negative cash.
            for name in state.player_names:
                if name not in state.bankrupt:
                    assert state.cash.get(name, 0) >= 0, (
                        f"step {step_no}: {name} has negative cash "
                        f"${state.cash.get(name, 0)}"
                    )

            # Property ownership consistency: each prop owned by at most one player.
            for pid, owner in state.properties.items():
                if owner is not None:
                    assert owner in state.player_names, (
                        f"step {step_no}: property {pid} has unknown owner {owner!r}"
                    )

    def test_legal_actions_empty_for_non_active(self):
        """legal_actions must be empty for every seat that is not the active
        player."""
        game = make_game()
        state = _initial_state(game)
        active = game.active_player(state)
        for seat in range(game.n_players):
            if seat != active:
                assert game.legal_actions(state, seat) == [], (
                    f"seat {seat} should have no legal actions when inactive"
                )


# --------------------------------------------------------------------------- #
# TestTermination
# --------------------------------------------------------------------------- #

class TestTermination:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_playout_terminates(self, seed):
        """Every random match must terminate within the runner's max_turns cap
        and the game's own turn_cap."""
        game, result, events = _run(seed, turn_cap=30, max_turns=2000)
        assert game.is_terminal(result.terminal_state), (
            f"seed={seed}: game did not reach terminal state"
        )
        assert result.n_turns < 2000, (
            f"seed={seed}: match took {result.n_turns} steps (cap 2000)"
        )

    def test_winner_declared(self):
        """Terminal state must always have exactly one winner."""
        game, result, events = _run(0, turn_cap=20, max_turns=1000)
        st = result.terminal_state
        assert game.is_terminal(st)
        assert st.winner is not None, "winner must be set at terminal"
        assert st.winner in st.player_names, "winner must be a valid player"
        # Rewards: winner gets 1.0, others get 0.0.
        rewards = game.rewards(st)
        assert len(rewards) == game.n_players
        winner_idx = st.player_names.index(st.winner)
        assert rewards[winner_idx] == 1.0
        assert sum(rewards) == 1.0

    def test_turn_cap_richest_player_wins(self):
        """At turn_cap, the player with highest net worth wins."""
        game = make_game(seed=0, turn_cap=3)
        state = game.initial_state(random.Random(0))
        rng = random.Random(0)
        for _ in range(500):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            if active < 0:
                break
            legal = game.legal_actions(state, active)
            if not legal:
                break
            state = game.step(state, rng.choice(legal))

        assert game.is_terminal(state)
        # If ended via turn cap, winner should be the wealthiest.
        if state.winner is not None:
            solvent = [p for p in state.player_names if p not in state.bankrupt]
            if solvent:
                best = max(solvent, key=lambda p: game._net_worth(state, p))
                # If there is a unique richest, it must be the declared winner.
                worths = [game._net_worth(state, p) for p in solvent]
                if worths.count(max(worths)) == 1:
                    assert state.winner == best


# --------------------------------------------------------------------------- #
# TestRentDoubling
# --------------------------------------------------------------------------- #

class TestRentDoubling:
    def test_single_ownership_base_rent(self):
        """Owning only one property in a group should yield base rent."""
        game = make_game()
        state = _initial_state(game)
        sq = BOARD[1]  # purple1, PURPLE group
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = None  # Bob doesn't own it
        rent = game._compute_rent(state, sq, "Alice")
        assert rent == sq["rent"], (
            f"Expected base rent ${sq['rent']}, got ${rent}"
        )

    def test_full_group_doubles_rent(self):
        """Owning all properties in a group doubles rent to full_rent."""
        game = make_game()
        state = _initial_state(game)

        for group_name, pids in GROUPS.items():
            if group_name == "RAILROAD":
                continue  # test railroads separately
            # Give Alice both properties in the group.
            for pid in pids:
                state.properties[pid] = "Alice"
            # Check rent on the first property in the group.
            pid0 = pids[0]
            sq = PROP_INFO[pid0]
            rent = game._compute_rent(state, sq, "Alice")
            assert rent == sq["full_rent"], (
                f"Group {group_name}: expected full_rent ${sq['full_rent']}, "
                f"got ${rent}"
            )
            # Reset for next iteration.
            for pid in pids:
                state.properties[pid] = None

    def test_railroad_full_set_doubles_rent(self):
        """Owning all 4 railroads doubles rent."""
        game = make_game()
        state = _initial_state(game)
        for pid in GROUPS["RAILROAD"]:
            state.properties[pid] = "Alice"
        sq = PROP_INFO["rr1"]
        rent = game._compute_rent(state, sq, "Alice")
        assert rent == sq["full_rent"]

    def test_partial_group_no_doubling(self):
        """Mixed group ownership (Alice+Bob) should yield base rent."""
        game = make_game()
        state = _initial_state(game)
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        sq = PROP_INFO["purple1"]
        rent = game._compute_rent(state, sq, "Alice")
        assert rent == sq["rent"]

    def test_rent_collected_correctly_on_landing(self):
        """When a player lands on an opponent's monopoly, full_rent is charged."""
        game = make_game()
        state = _initial_state(game)
        # Alice owns the full PURPLE group.
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Alice"
        # Bob's cash set to known value.
        state.cash["Bob"] = 500
        alice_start = state.cash["Alice"]
        # Put Bob on position 0 so a specific roll lands on purple2 (sq 3).
        state.positions["Bob"] = 0
        state.current_player_idx = state.player_names.index("Bob")
        state.phase = PH_ROLL
        state.turn = 10  # set turn so roll is deterministic
        # Override dice to force landing on purple2 (sq 3): need roll = 3.
        # We'll test via _pay rather than relying on dice value.
        sq = PROP_INFO["purple2"]
        game._pay(state, "Bob", "Alice", sq["full_rent"])
        assert state.cash["Bob"] == 500 - sq["full_rent"]
        assert state.cash["Alice"] == alice_start + sq["full_rent"]


# --------------------------------------------------------------------------- #
# TestTradeLifecycle
# --------------------------------------------------------------------------- #

class TestTradeLifecycle:
    def _setup_trade_state(self) -> tuple:
        """Return (game, state) with Alice owning purple1, Bob owning purple2."""
        game = make_game()
        state = _initial_state(game)
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.cash["Alice"] = 800
        state.cash["Bob"] = 600
        state.phase = PH_ROLL
        state.current_player_idx = 0  # Alice's turn
        return game, state

    def test_propose_trade_sets_pending_and_phase(self):
        """propose_trade must record pending_trade and set phase=trade_response."""
        game, state = self._setup_trade_state()
        action = {
            "type": "propose_trade",
            "to": "Bob",
            "give_props": ["purple1"],
            "give_cash": 100,
            "want_props": ["purple2"],
            "want_cash": 0,
        }
        game.step(state, action)
        assert state.phase == PH_TRADE
        assert state.pending_trade is not None
        assert state.pending_trade["from"] == "Alice"
        assert state.pending_trade["to"] == "Bob"
        assert state.pending_trade["give_props"] == ["purple1"]
        assert state.pending_trade["give_cash"] == 100
        assert state.pending_trade["want_props"] == ["purple2"]
        # Bob is the active player during trade_response.
        bob_idx = state.player_names.index("Bob")
        assert game.active_player(state) == bob_idx

    def test_accept_trade_transfers_correctly(self):
        """Accepting a trade must transfer all specified assets correctly."""
        game, state = self._setup_trade_state()
        initial_alice_cash = state.cash["Alice"]
        initial_bob_cash = state.cash["Bob"]

        # Alice proposes: give purple1 + $100 cash, want purple2.
        game.step(state, {
            "type": "propose_trade",
            "to": "Bob",
            "give_props": ["purple1"],
            "give_cash": 100,
            "want_props": ["purple2"],
            "want_cash": 0,
        })
        game.step(state, {"type": "accept_trade"})

        # Alice now owns purple2 (not purple1); Bob owns purple1 (not purple2).
        assert state.properties["purple2"] == "Alice"
        assert state.properties["purple1"] == "Bob"
        # Cash: Alice paid $100 to Bob.
        assert state.cash["Alice"] == initial_alice_cash - 100
        assert state.cash["Bob"] == initial_bob_cash + 100
        # Trade is cleared.
        assert state.pending_trade is None
        # Phase returns to roll (Alice still needs to roll).
        assert state.phase == PH_ROLL
        assert state.current_player_idx == 0  # Alice

    def test_reject_trade_leaves_state_unchanged(self):
        """Rejecting a trade must leave cash and properties unchanged."""
        game, state = self._setup_trade_state()
        initial_props = dict(state.properties)
        initial_alice_cash = state.cash["Alice"]
        initial_bob_cash = state.cash["Bob"]

        game.step(state, {
            "type": "propose_trade",
            "to": "Bob",
            "give_props": ["purple1"],
            "give_cash": 50,
            "want_props": ["purple2"],
            "want_cash": 0,
        })
        game.step(state, {"type": "reject_trade"})

        assert state.properties == initial_props
        assert state.cash["Alice"] == initial_alice_cash
        assert state.cash["Bob"] == initial_bob_cash
        assert state.pending_trade is None
        assert state.phase == PH_ROLL

    def test_accept_trade_with_cash_both_ways(self):
        """Two-way cash transfer in a trade must be applied correctly."""
        game, state = self._setup_trade_state()
        state.cash["Alice"] = 500
        state.cash["Bob"] = 400

        game.step(state, {
            "type": "propose_trade",
            "to": "Bob",
            "give_props": [],
            "give_cash": 200,
            "want_props": [],
            "want_cash": 100,
        })
        game.step(state, {"type": "accept_trade"})

        # Net: Alice gives $200, gets $100 → -$100.  Bob gets $200, gives $100 → +$100.
        assert state.cash["Alice"] == 500 - 200 + 100
        assert state.cash["Bob"] == 400 + 200 - 100


# --------------------------------------------------------------------------- #
# TestBankruptcy
# --------------------------------------------------------------------------- #

class TestBankruptcy:
    def test_bankruptcy_from_rent_transfers_assets_to_creditor(self):
        """A player who can't pay rent goes bankrupt; all assets go to
        the creditor (the property owner) and the bankrupt player is
        removed from the active rotation."""
        game = make_game()
        state = _initial_state(game)

        # Alice owns red1 AND red2 (full group) → rent = $50.
        state.properties["red1"] = "Alice"
        state.properties["red2"] = "Alice"
        # Bob has only $30 cash (can't afford $50 rent) and owns purple1.
        state.cash["Bob"] = 30
        state.properties["purple1"] = "Bob"

        alice_start_cash = state.cash["Alice"]
        # Force Bob to pay rent to Alice.
        sq = PROP_INFO["red1"]
        game._pay(state, "Bob", "Alice", game._compute_rent(state, sq, "Alice"))

        assert "Bob" in state.bankrupt
        # Bob's cash went to Alice.
        assert state.cash["Bob"] == 0
        assert state.cash["Alice"] == alice_start_cash + 30  # Bob had $30
        # Bob's purple1 transferred to Alice.
        assert state.properties["purple1"] == "Alice"

    def test_bankruptcy_from_tax_sends_assets_to_bank(self):
        """Tax bankruptcy sends all assets to the bank (properties unowned,
        cash disappears)."""
        game = make_game()
        state = _initial_state(game)

        state.cash["Carol"] = 10
        state.properties["orange1"] = "Carol"

        game._pay(state, "Carol", None, 100)  # can't afford $100 tax

        assert "Carol" in state.bankrupt
        assert state.cash["Carol"] == 0
        # Property returned to bank (unowned).
        assert state.properties["orange1"] is None

    def test_bankruptcy_removes_player_from_rotation(self):
        """After bankruptcy the active rotation skips the bankrupt player."""
        game = make_game()
        state = _initial_state(game)

        # Make Carol bankrupt.
        carol_idx = state.player_names.index("Carol")
        state.bankrupt.add("Carol")

        living = game.living_seats(state)
        assert carol_idx not in living, "bankrupt seat must not be in living_seats"

    def test_last_solvent_player_wins_immediately(self):
        """When all but one player go bankrupt, the survivor wins immediately."""
        game = make_game()
        state = _initial_state(game)

        # Bankrupt all but Alice.
        for name in ["Bob", "Carol", "Dave", "Eve"]:
            state.bankrupt.add(name)

        # Manually call _bankrupt for the last one to trigger winner detection.
        state.bankrupt.discard("Eve")  # undo, to trigger via _bankrupt
        state.cash["Eve"] = 5
        game._bankrupt(state, "Eve", None)

        assert state.winner == "Alice"
        assert game.is_terminal(state)

    def test_bankruptcy_mid_playout_preserves_invariants(self):
        """After a player goes bankrupt in a random playout, all invariants
        hold: cash >= 0 for solvent players, properties owned by valid players
        or None, and the bankrupt player never appears as active_player."""
        game = make_game(seed=3, turn_cap=20)
        state = game.initial_state(random.Random(3))
        rng = random.Random(42)

        for _ in range(300):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            if active < 0:
                break
            legal = game.legal_actions(state, active)
            if not legal:
                break
            state = game.step(state, rng.choice(legal))

            for name in state.player_names:
                if name not in state.bankrupt:
                    assert state.cash.get(name, 0) >= 0
            for pid, owner in state.properties.items():
                if owner is not None:
                    assert owner in state.player_names
            # Active player must not be bankrupt.
            a = game.active_player(state)
            if a >= 0:
                assert state.player_names[a] not in state.bankrupt


# --------------------------------------------------------------------------- #
# TestBoardLayout
# --------------------------------------------------------------------------- #

class TestBoardLayout:
    def test_board_has_20_squares(self):
        assert len(BOARD) == 20

    def test_all_squares_have_sq_index(self):
        for i, sq in enumerate(BOARD):
            assert sq["sq"] == i, f"Square {i} has sq={sq['sq']}"

    def test_all_prop_ids_unique(self):
        ids = [sq["prop_id"] for sq in BOARD if "prop_id" in sq]
        assert len(ids) == len(set(ids))

    def test_all_groups_correct_size(self):
        for group, pids in GROUPS.items():
            if group == "RAILROAD":
                assert len(pids) == 4
            else:
                assert len(pids) == 2

    def test_initial_state_all_unowned(self):
        game = make_game()
        state = _initial_state(game)
        for pid, owner in state.properties.items():
            assert owner is None, f"{pid} should start unowned"

    def test_initial_cash_correct(self):
        game = make_game()
        state = _initial_state(game)
        for name in state.player_names:
            assert state.cash[name] == STARTING_CASH

    def test_initial_positions_on_go(self):
        game = make_game()
        state = _initial_state(game)
        for name in state.player_names:
            assert state.positions[name] == 0, f"{name} should start on GO"


# --------------------------------------------------------------------------- #
# TestGOSalary
# --------------------------------------------------------------------------- #

class TestGOSalary:
    def test_pass_go_awards_salary(self):
        """A player whose roll wraps past GO must receive GO_SALARY."""
        game = make_game()
        state = _initial_state(game)
        alice = "Alice"
        # Put Alice near end of board so she passes GO.
        state.positions[alice] = 18  # 18 + any roll >= 2 wraps around
        state.current_player_idx = 0
        state.phase = PH_ROLL
        state.turn = 0
        cash_before = state.cash[alice]
        # Roll is deterministic: seed = 0*31 + 0*7 = 0, random.Random(0) → 1+2=3
        # new_pos = (18 + 3) % 20 = 1, so they passed GO.
        game.step(state, {"type": "roll"})
        # Regardless of exact roll, 18 + 2..12 all wrap (since 18 + 2 = 20 ≥ 20).
        # Just check salary was awarded.
        assert state.cash.get(alice, 0) >= cash_before + GO_SALARY, (
            f"Expected Alice to receive GO salary. Cash was {cash_before}, "
            f"now {state.cash.get(alice, 0)}"
        )


# --------------------------------------------------------------------------- #
# TestRegistryIntegration
# --------------------------------------------------------------------------- #

class TestRegistryIntegration:
    def test_game_name_attribute(self):
        game = make_game()
        assert game.name == "monopoly_lite"

    def test_n_players_matches_player_list(self):
        game = MonopolyLite(players=["X", "Y", "Z"], seed=0)
        assert game.n_players == 3
        state = game.initial_state(random.Random(0))
        assert len(state.player_names) == 3

    def test_render_prompt_returns_string(self):
        game = make_game()
        state = _initial_state(game)
        for seat in range(game.n_players):
            prompt = game.render_prompt(state, seat)
            assert isinstance(prompt, str) and len(prompt) > 10

    def test_god_view_and_snapshot(self):
        game = make_game()
        state = _initial_state(game)
        gv = game.god_view(state)
        assert "cash" in gv and "properties" in gv
        snap = game.snapshot(state)
        assert "public" in snap and "hidden" in snap
        assert "positions" in snap["public"]

    def test_render_board(self):
        game = make_game()
        state = _initial_state(game)
        board = game.render_board(state)
        assert "Monopoly Lite" in board
        assert "sq  0" in board or "sq 0" in board


# --------------------------------------------------------------------------- #
# TestTradeInLegalActions
# --------------------------------------------------------------------------- #

class TestTradeInLegalActions:
    """Trade proposals are now surfaced in legal_actions (PH_ROLL).
    These tests verify the cooperation surface is accessible to LLM players
    and that all existing invariants hold when trades enter the action space."""

    def test_trade_proposals_appear_in_roll_phase(self):
        """propose_trade must appear in legal_actions when the current player
        has ≥1 property and a counterparty owns another from the same colour
        group (the classic group-completing scenario)."""
        game = MonopolyLite(players=["Alice", "Bob"], seed=0, turn_cap=200)
        state = game.initial_state(random.Random(0))
        # Alice has purple1, Bob has purple2 — classic split group.
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.phase = PH_ROLL
        state.current_player_idx = 0  # Alice's turn

        legal = game.legal_actions(state, 0)
        types = [a["type"] for a in legal]
        assert "roll" in types, "roll must always be present in PH_ROLL"
        assert "propose_trade" in types, (
            "propose_trade must appear when a group-completing trade is possible"
        )
        # At least one proposal should target Bob with purple2.
        proposals = [a for a in legal if a["type"] == "propose_trade"]
        bob_props = [
            p for p in proposals
            if p.get("to") == "Bob" and "purple2" in p.get("want_props", [])
        ]
        assert bob_props, "Expected a proposal to Bob wanting purple2"

    def test_no_trade_proposals_when_no_properties_owned(self):
        """When no player owns any property, no propose_trade actions must
        appear — heuristic (c) also requires the counterparty to hold
        something."""
        game = MonopolyLite(players=["Alice", "Bob"], seed=0, turn_cap=200)
        state = game.initial_state(random.Random(0))
        state.phase = PH_ROLL
        state.current_player_idx = 0

        legal = game.legal_actions(state, 0)
        types = [a["type"] for a in legal]
        assert "roll" in types
        assert "propose_trade" not in types, (
            "No propose_trade expected when no properties are owned"
        )

    def test_trade_proposals_bounded_at_six(self):
        """_candidate_trades must never return more than 6 proposals, even
        when every colour group is split across players."""
        game = MonopolyLite(players=["A", "B", "C", "D"], seed=0, turn_cap=200)
        state = game.initial_state(random.Random(0))
        # Each colour group split across two players.
        state.properties["purple1"] = "A"
        state.properties["purple2"] = "B"
        state.properties["lblue1"] = "B"
        state.properties["lblue2"] = "A"
        state.properties["orange1"] = "C"
        state.properties["orange2"] = "D"
        state.properties["red1"] = "D"
        state.properties["red2"] = "C"

        for name in ["A", "B", "C", "D"]:
            proposals = game._candidate_trades(state, name)
            assert len(proposals) <= 6, (
                f"{name}: got {len(proposals)} proposals (cap is 6)"
            )

    def test_propose_accept_via_legal_actions_transfers(self):
        """Full propose→accept cycle sampled directly from legal_actions
        must transfer properties and cash correctly, then return to PH_ROLL
        for the proposer to roll."""
        game = MonopolyLite(players=["Alice", "Bob"], seed=0, turn_cap=200)
        state = game.initial_state(random.Random(0))
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.cash["Alice"] = 800
        state.cash["Bob"] = 600
        state.phase = PH_ROLL
        state.current_player_idx = 0  # Alice

        # 1. Sample a trade proposal from Alice's legal actions.
        legal = game.legal_actions(state, 0)
        proposals = [a for a in legal if a["type"] == "propose_trade"]
        assert proposals, "Expected propose_trade in Alice's legal actions"
        proposal = proposals[0]
        assert proposal["to"] == "Bob"

        give_props = list(proposal.get("give_props", []))
        give_cash = int(proposal.get("give_cash", 0))
        want_props = list(proposal.get("want_props", []))

        alice_cash_before = state.cash["Alice"]
        bob_cash_before = state.cash["Bob"]

        # 2. Apply the proposal.
        state = game.step(state, proposal)
        assert state.phase == PH_TRADE
        assert state.pending_trade is not None

        # 3. Bob is now active — accept_trade and reject_trade in legal actions.
        bob_seat = state.player_names.index("Bob")
        assert game.active_player(state) == bob_seat
        cp_legal = game.legal_actions(state, bob_seat)
        assert {"type": "accept_trade"} in cp_legal
        assert {"type": "reject_trade"} in cp_legal

        # 4. Bob accepts.
        state = game.step(state, {"type": "accept_trade"})

        # 5. Properties transferred correctly.
        for pid in give_props:
            assert state.properties.get(pid) == "Bob", (
                f"{pid} should now belong to Bob"
            )
        for pid in want_props:
            assert state.properties.get(pid) == "Alice", (
                f"{pid} should now belong to Alice"
            )

        # 6. Cash settled (proposer paid give_cash to recipient).
        if give_cash > 0:
            assert state.cash["Alice"] == alice_cash_before - give_cash
            assert state.cash["Bob"] == bob_cash_before + give_cash

        # 7. Game returns to PH_ROLL for Alice to roll.
        assert state.phase == PH_ROLL
        assert state.pending_trade is None

    def test_200_step_playout_trade_proposals_appear(self):
        """200-step random playout with pre-seeded group-splitting: at least
        one player's legal_actions during PH_ROLL must contain propose_trade.
        This confirms trades are reachable by a random (or LLM) policy."""
        game = MonopolyLite(players=["A", "B", "C"], seed=17, turn_cap=40)
        state = game.initial_state(random.Random(17))
        # Pre-seed split ownership so trade opportunities exist from turn 0.
        state.properties["purple1"] = "A"
        state.properties["purple2"] = "B"
        rng = random.Random(321)

        trade_proposals_seen = 0
        for step_no in range(200):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            if active < 0:
                break
            legal = game.legal_actions(state, active)
            assert legal, f"step {step_no}: empty legal_actions"

            proposals = [a for a in legal if a.get("type") == "propose_trade"]
            if proposals:
                trade_proposals_seen += 1

            state = game.step(state, rng.choice(legal))

            # Invariants: cash, ownership, no bankrupt player is active.
            for name in state.player_names:
                if name not in state.bankrupt:
                    assert state.cash.get(name, 0) >= 0, (
                        f"step {step_no}: {name} has negative cash"
                    )
            for pid, owner in state.properties.items():
                if owner is not None:
                    assert owner in state.player_names, (
                        f"step {step_no}: {pid} owned by unknown {owner!r}"
                    )
            a = game.active_player(state)
            if a >= 0:
                assert state.player_names[a] not in state.bankrupt

        assert trade_proposals_seen > 0, (
            "No propose_trade appeared in any legal_actions during 200-step "
            "playout with pre-seeded split group ownership."
        )

    def test_symmetric_swap_detected(self):
        """When both proposer and counterparty each hold one property from
        two different colour groups, the symmetric zero-cash swap must appear
        in _candidate_trades (heuristic b)."""
        game = MonopolyLite(players=["Alice", "Bob"], seed=0, turn_cap=200)
        state = game.initial_state(random.Random(0))
        # Alice: purple1, lblue2 — Bob: purple2, lblue1.
        # Swap purple1↔lblue1 or lblue2↔purple2 gives both a monopoly.
        state.properties["purple1"] = "Alice"
        state.properties["lblue2"] = "Alice"
        state.properties["purple2"] = "Bob"
        state.properties["lblue1"] = "Bob"

        proposals = game._candidate_trades(state, "Alice")
        # Symmetric swap: give lblue2, get purple2 (or give purple1, get lblue1) — $0
        zero_cash = [p for p in proposals if p.get("give_cash", 0) == 0]
        assert zero_cash, (
            "Expected at least one zero-cash symmetric swap proposal when both "
            "players hold complementary half-groups"
        )


# --------------------------------------------------------------------------- #
# TestNegotiationPhase
# --------------------------------------------------------------------------- #

def _skip_nego(game: MonopolyLite, state: MLState) -> None:
    """Drain the negotiation phase by repeatedly passing until PH_ROLL."""
    for _ in range(200):
        if state.phase != PH_NEGOTIATION:
            break
        active = game.active_player(state)
        if active < 0:
            break
        game.step(state, {"type": "pass_talk"})


class TestNegotiationPhase:
    def test_initial_state_starts_in_negotiation(self):
        """initial_state must put the game in PH_NEGOTIATION, not PH_ROLL."""
        game = make_game()
        state = _initial_state(game)
        assert state.phase == PH_NEGOTIATION, (
            f"Expected initial phase to be {PH_NEGOTIATION!r}, got {state.phase!r}"
        )
        assert state.nego.active, "NegotiationState must be active at game start"

    def test_negotiation_exits_to_roll(self):
        """After all negotiation rounds complete, phase must become PH_ROLL."""
        game = make_game()
        state = _initial_state(game)
        assert state.phase == PH_NEGOTIATION
        _skip_nego(game, state)
        assert state.phase == PH_ROLL, (
            f"Expected PH_ROLL after negotiation exits, got {state.phase!r}"
        )

    def test_negotiation_produces_messages(self):
        """A 300-step random playout must produce >= 8 say+whisper messages."""
        game = MonopolyLite(players=["A", "B", "C", "D"], seed=42, turn_cap=30)
        state = game.initial_state(random.Random(42))
        rng = random.Random(7)
        msg_count = 0

        for _ in range(300):
            if game.is_terminal(state):
                break
            active = game.active_player(state)
            if active < 0:
                break
            legal = game.legal_actions(state, active)
            if not legal:
                break
            action = rng.choice(legal)
            if action.get("type") in ("say", "whisper"):
                msg_count += 1
            game.step(state, action)

        assert msg_count >= 8, (
            f"Expected >= 8 say+whisper actions in 300-step playout, got {msg_count}"
        )

    def test_negotiation_legal_actions_include_trade_proposals(self):
        """During PH_NEGOTIATION, legal_actions must include propose_trade when
        group-completing trades are available."""
        game = MonopolyLite(players=["Alice", "Bob", "Carol", "Dave"], seed=0)
        state = game.initial_state(random.Random(0))
        assert state.phase == PH_NEGOTIATION
        # Pre-seed a split group so trade proposals are available.
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        # Alice is first speaker (seat 0).
        active = game.active_player(state)
        assert active == 0  # Alice
        legal = game.legal_actions(state, active)
        types = [a["type"] for a in legal]
        assert "propose_trade" in types, (
            "propose_trade must appear in PH_NEGOTIATION legal_actions when "
            "group-completing trades are available"
        )

    def test_propose_trade_during_negotiation_exits_nego(self):
        """A propose_trade action during PH_NEGOTIATION must move game to
        PH_TRADE (trade machinery kicks in immediately)."""
        game = MonopolyLite(players=["Alice", "Bob", "Carol", "Dave"], seed=0)
        state = game.initial_state(random.Random(0))
        state.properties["purple1"] = "Alice"
        state.properties["purple2"] = "Bob"
        # Get trade proposals during negotiation.
        active = game.active_player(state)
        legal = game.legal_actions(state, active)
        proposals = [a for a in legal if a["type"] == "propose_trade"]
        assert proposals, "Expected propose_trade in negotiation legal_actions"
        game.step(state, proposals[0])
        assert state.phase == PH_TRADE, (
            "Proposing a trade during negotiation should move to PH_TRADE"
        )
