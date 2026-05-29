"""Tests for the full-map Risk game (``play/games/risk_lite.py``, Unit 3).

Covers the §5.3 feature-parity checklist points that are mechanically
verifiable without an LLM:

  * a 4-player RandomPlayer match terminates with a valid winner within the
    hard round cap and well within ``max_turns=400``;
  * the full 42-territory map is wired in (territory distribution covers all
    42, owners are valid seats);
  * reinforcements = ``max(3, territories // 3)`` plus continent bonuses, and
    the continent bonus is applied correctly when a player holds a whole
    continent;
  * Risk-card set trading escalates per ``SET_VALUES`` then ``+5``;
  * attacking an active ally with ``enforce_alliances=False`` is LEGAL, emits a
    ``betrayed`` alliance_event, and breaks the alliance;
  * attacking an active ally with ``enforce_alliances=True`` is ILLEGAL (filtered
    from ``legal_actions`` and rejected by ``parse_action``);
  * ``combat`` observation records are emitted for attacks;
  * the negotiation sub-phase is reachable and the message/alliance lifecycle
    runs end-to-end under RandomPlayer.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from game_theory_llm.play import run_match
from game_theory_llm.play.games import RiskLite
from game_theory_llm.play.games.risk_lite import (
    RiskLite as RiskLiteClass,
    RLState,
    PH_ATTACK,
    PH_NEGOTIATION,
    PH_DEPLOY,
    PH_FORTIFY,
    PH_TERMINAL,
    MAX_ROUNDS,
    N_TERRITORIES,
    ALLIANCE_KINDS,
)
from game_theory_llm.play.config import GameConfig
from game_theory_llm.play.alliances import Alliance
from game_theory_llm.play.maps.risk_map import (
    TERRITORIES, ADJ, CONTINENTS, SET_VALUES, set_value,
)
from game_theory_llm.play.players import RandomPlayer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _run(seed: int, *, config=None, max_turns: int = 400):
    game = RiskLite(config) if config is not None else RiskLite()
    players = [RandomPlayer(seed=seed * 31 + i) for i in range(game.n_players)]
    tmp = tempfile.mkdtemp()
    log_path = Path(tmp) / f"risk_{seed}.jsonl"
    result = run_match(game, players, seed=seed, log_path=log_path,
                       max_turns=max_turns)
    events = [json.loads(l) for l in log_path.read_text().splitlines()]
    return game, result, events


def _allied_attack_state(enforce: bool):
    """Build a state where P0 (src) is adjacent to an enemy P1 territory and
    P0/P1 share an *active* alliance, with P0 to act in the attack phase."""
    game = RiskLite(GameConfig(enforce_alliances=enforce))
    st = game.initial_state(random.Random(0))
    src = 0
    st.owner[src] = 0
    st.armies[src] = 5
    nb = sorted(ADJ[src])[0]
    st.owner[nb] = 1
    st.armies[nb] = 1
    al = Alliance(id=0, members=[0, 1], proposer=0, kind="mutual_defense",
                  terms={}, status="active", proposed_turn=0, accepted_turn=0)
    st.alli.alliances[0] = al
    st.alli.next_id = 1
    st.phase = PH_ATTACK
    st.current_player = 0
    return game, st, src, nb


# --------------------------------------------------------------------------- #
# Basic identity / map wiring
# --------------------------------------------------------------------------- #

def test_name_and_default_players():
    game = RiskLite()
    assert game.name == "risk"
    assert game.n_players == 4
    # No-arg constructor must work (INV-4) and produce a default config.
    assert isinstance(game.config, GameConfig)


def test_full_42_territory_distribution():
    game = RiskLite()
    st = game.initial_state(random.Random(3))
    assert len(st.owner) == N_TERRITORIES == 42
    assert len(st.armies) == 42
    # Every territory owned by a valid seat; every seat owns >= 1 territory.
    assert all(0 <= o < game.n_players for o in st.owner)
    owned_counts = {p: sum(1 for o in st.owner if o == p)
                    for p in range(game.n_players)}
    assert sum(owned_counts.values()) == 42
    for p in range(game.n_players):
        assert owned_counts[p] >= 1
    # Every owned territory has at least one army.
    assert all(a >= 1 for a in st.armies)


def test_starts_in_negotiation():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    assert st.phase == PH_NEGOTIATION
    # The negotiation sub-phase is active with a non-empty speak queue.
    assert st.nego.active
    assert st.nego.speak_queue


# --------------------------------------------------------------------------- #
# Termination (the headline requirement)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
def test_match_terminates_with_valid_winner(seed):
    game, result, events = _run(seed)
    st = result.terminal_state
    assert game.is_terminal(st)
    assert st.phase == PH_TERMINAL
    # A valid winner is declared.
    assert st.winner is not None
    assert 0 <= st.winner < game.n_players
    # Winner is alive.
    assert not st.eliminated[st.winner]
    # Exactly one seat gets reward 1.0.
    assert sum(int(r > 0) for r in result.rewards) == 1
    assert result.rewards[st.winner] == 1.0
    # Comfortably within the runner cap.
    assert result.n_turns < 400
    # terminal record is last.
    assert events[-1]["type"] == "terminal"


def test_winner_has_most_territories_at_cap():
    """When the match ends at the round cap (the common RandomPlayer outcome),
    the winner must hold the most territories (ties -> lowest seat)."""
    game, result, _ = _run(0)
    st = result.terminal_state
    if st.round_no > MAX_ROUNDS:
        counts = {p: sum(1 for o in st.owner if o == p)
                  for p in range(game.n_players) if not st.eliminated[p]}
        best = max(counts, key=lambda p: (counts[p], -p))
        assert st.winner == best
        assert counts[st.winner] == max(counts.values())


# --------------------------------------------------------------------------- #
# Reinforcements + continent bonuses
# --------------------------------------------------------------------------- #

def test_reinforcement_base_formula():
    game = RiskLite()
    st = game.initial_state(random.Random(1))
    # Strip P0 of every continent bonus by ensuring it does not hold any whole
    # continent: give P0 exactly 9 territories, none forming a full continent.
    # Easiest: recompute the base via the helper against a hand-built owner map.
    for t in range(N_TERRITORIES):
        st.owner[t] = 1
    # Give P0 one territory from each of three different continents (no whole
    # continent), 9 territories total but spread so no continent completes.
    spread = [0, 9, 13, 20, 26, 38, 1, 10, 14]   # mixed continents
    for t in spread:
        st.owner[t] = 0
    terr = sum(1 for o in st.owner if o == 0)
    assert game._continent_bonus(st, 0) == 0  # no whole continent held
    assert game._reinforcements_for(st, 0) == max(3, terr // 3)


def test_continent_bonus_applied():
    game = RiskLite()
    st = game.initial_state(random.Random(2))
    for cont, (members, value) in CONTINENTS.items():
        # Reset and give P0 exactly this continent's members (only).
        for t in range(N_TERRITORIES):
            st.owner[t] = 1
        for t in members:
            st.owner[t] = 0
        terr = sum(1 for o in st.owner if o == 0)
        expected = max(3, terr // 3) + value
        assert game._reinforcements_for(st, 0) == expected, (
            f"{cont}: expected {expected}")


# --------------------------------------------------------------------------- #
# Risk-card set trading
# --------------------------------------------------------------------------- #

def test_set_trade_escalating_values():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    # Give P0 a one-of-each set repeatedly and confirm escalating bonuses.
    awarded = []
    for i in range(8):
        st.cards[0] = ["infantry", "cavalry", "artillery"]
        bonus = game._trade_set(st, 0)
        awarded.append(bonus)
    # First six follow the table, then +5 each.
    assert awarded[:6] == SET_VALUES
    assert awarded[6] == SET_VALUES[-1] + 5
    assert awarded[7] == SET_VALUES[-1] + 10
    assert awarded == [set_value(i) for i in range(8)]


def test_three_of_a_kind_is_a_valid_set():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    st.cards[0] = ["cavalry", "cavalry", "cavalry"]
    assert game._find_set(st.cards[0]) is not None
    bonus = game._trade_set(st, 0)
    assert bonus == SET_VALUES[0]
    assert st.cards[0] == []


def test_no_set_no_trade():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    st.cards[0] = ["infantry", "infantry"]
    assert game._find_set(st.cards[0]) is None
    assert game._trade_set(st, 0) == 0


# --------------------------------------------------------------------------- #
# Alliances: betrayal (enforce off) vs prohibition (enforce on)
# --------------------------------------------------------------------------- #

def test_attacking_ally_betrays_and_breaks_when_not_enforced():
    game, st, src, nb = _allied_attack_state(enforce=False)
    # The attack on the ally is LEGAL.
    legal = game.legal_actions(st, 0)
    attacks = [a for a in legal
               if a.get("type") == "attack" and a["src"] == src and a["dst"] == nb]
    assert attacks, "attacking an ally must be legal when enforce_alliances=False"
    # Resolving it records a betrayal and breaks the alliance.
    game.step(st, {"type": "attack", "src": src, "dst": nb})
    betrayed = [e for e in st.alli.events if e["event"] == "betrayed"]
    assert len(betrayed) == 1
    ev = betrayed[0]
    assert ev["actor"] == 0
    assert 1 in ev.get("counterparty", [])    # P1 is the betrayed victim
    assert ev["action_ref"]["type"] == "attack"
    assert st.alli.alliances[0].status == "broken"
    assert st.alli.alliances[0].broken_by == 0


def test_attacking_ally_is_illegal_when_enforced():
    game, st, src, nb = _allied_attack_state(enforce=True)
    # Filtered out of legal_actions.
    legal = game.legal_actions(st, 0)
    attacks = [a for a in legal
               if a.get("type") == "attack" and a["src"] == src and a["dst"] == nb]
    assert not attacks, "attacking an ally must be filtered when enforce_alliances=True"
    # Rejected by parse_action.
    with pytest.raises(Exception):
        game.parse_action(st, 0, f"<attack src={src} dst={nb}></attack>")
    # No betrayal event recorded.
    assert not [e for e in st.alli.events if e["event"] == "betrayed"]


def test_attacking_non_ally_is_legal_under_enforcement():
    """Enforcement only blocks attacks against active allies, not all enemies."""
    game = RiskLite(GameConfig(enforce_alliances=True))
    st = game.initial_state(random.Random(0))
    src = 0
    st.owner[src] = 0
    st.armies[src] = 5
    nb = sorted(ADJ[src])[0]
    st.owner[nb] = 2          # P2, not allied
    st.armies[nb] = 1
    st.phase = PH_ATTACK
    st.current_player = 0
    legal = game.legal_actions(st, 0)
    attacks = [a for a in legal
               if a.get("type") == "attack" and a["src"] == src and a["dst"] == nb]
    assert attacks, "attacking a non-ally must remain legal under enforcement"


# --------------------------------------------------------------------------- #
# Elimination captures cards
# --------------------------------------------------------------------------- #

def test_elimination_captures_cards():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    # Make P1 hold exactly one territory (its last), adjacent to a strong P0
    # stack, and give P1 two cards.
    src = 0
    st.owner[src] = 0
    st.armies[src] = 20            # huge stack -> will clear the 1-army defender
    nb = sorted(ADJ[src])[0]
    # Wipe P1 everywhere, then leave it only nb.
    for t in range(N_TERRITORIES):
        if st.owner[t] == 1:
            st.owner[t] = 2
    st.owner[nb] = 1
    st.armies[nb] = 1
    st.cards[1] = ["infantry", "cavalry"]
    st.cards[0] = []
    st.phase = PH_ATTACK
    st.current_player = 0
    # Attack until capture (deterministic combat; 20 vs 1 should clear quickly).
    for _ in range(20):
        if st.owner[nb] == 0:
            break
        game.step(st, {"type": "attack", "src": src, "dst": nb})
    assert st.owner[nb] == 0, "P0 should have captured P1's last territory"
    assert st.eliminated[1], "P1 should be eliminated"
    # P0 seized P1's two cards.
    assert "infantry" in st.cards[0] and "cavalry" in st.cards[0]
    assert st.cards[1] == []


# --------------------------------------------------------------------------- #
# combat observation records
# --------------------------------------------------------------------------- #

def test_combat_observation_records_emitted():
    # Find a seed whose RandomPlayer match contains at least one attack.
    for seed in range(6):
        _, _, events = _run(seed)
        combat = [e for e in events
                  if e.get("type") == "observation"
                  and e["obs"].get("type") == "combat"]
        if combat:
            c = combat[0]["obs"]
            for key in ("src", "dst", "atk_rolls", "def_rolls",
                        "atk_lost", "def_lost", "captured"):
                assert key in c, f"combat obs missing {key}"
            return
    pytest.fail("no combat observation records across 6 random matches")


def test_conquest_win_combat_obs_reports_the_winning_capture():
    """M2: the decisive conquest-winning capture must produce a fresh combat
    observation with captured:True and the correct src/dst — not a stale
    record from a previous (non-capturing) attack."""
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    src, nb = 0, sorted(ADJ[0])[0]
    # Give the whole board to P0 except a single P1 territory (nb) adjacent to a
    # large P0 stack at src. One capture of nb -> P0 controls all 42.
    for t in range(N_TERRITORIES):
        st.owner[t] = 0
        st.armies[t] = 1
    st.owner[nb] = 1
    st.armies[nb] = 1
    st.armies[src] = 30          # overwhelming -> certain capture
    st.eliminated = [False] * game.n_players
    st.phase = PH_ATTACK
    st.current_player = 0
    # Seed last_attack_roll with a STALE, non-capturing record to prove the
    # observation is rebuilt from the just-resolved winning attack, not cached.
    st.last_attack_roll = {
        "src": 5, "dst": 6, "atk_rolls": [1], "def_rolls": [6],
        "atk_lost": 1, "def_lost": 0, "captured": False,
    }
    prev = st  # runner mutates in place; prev_state is the same object
    action = {"type": "attack", "src": src, "dst": nb}
    new_state = game.step(st, action)
    # The attack won the game.
    assert new_state.phase == PH_TERMINAL
    assert new_state.winner == 0
    # The combat observation reflects the winning capture, not the stale record.
    obs = game.observations(prev, new_state, action, actor=0)
    combat = [o for o in obs if o.payload.get("type") == "combat"]
    assert combat, "winning capture must still emit a combat observation"
    c = combat[0].payload
    assert c["captured"] is True
    assert c["src"] == src and c["dst"] == nb
    # And the underlying record matches (not the seeded 5->6 stale one).
    assert new_state.last_attack_roll["src"] == src
    assert new_state.last_attack_roll["dst"] == nb
    assert new_state.last_attack_roll["captured"] is True


def test_elimination_forced_trade_armies_are_placed_not_lost():
    """m2: when an elimination capture pushes the capturer to >= 5 cards, the
    forced card-set trade's bonus armies must be placed on the captured
    territory immediately — _next_player would otherwise zero them out."""
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    src, nb = 0, sorted(ADJ[0])[0]
    # P0 attacks with an overwhelming stack to capture P1's LAST territory (nb),
    # eliminating P1. Give the rest of the board to P2 so this is NOT a
    # conquest win (so we exercise the place-immediately path, not the win
    # early-return). P0 already holds 4 cards; P1's seized card makes 5 ->
    # forced trade.
    for t in range(N_TERRITORIES):
        st.owner[t] = 2
        st.armies[t] = 1
    st.owner[src] = 0
    st.armies[src] = 30
    st.owner[nb] = 1
    st.armies[nb] = 1
    st.eliminated = [False] * game.n_players
    # P0 holds a ready one-of-each set plus one extra card; P1 holds one card so
    # that after seizing, P0 has >= 5 cards -> at least one forced trade fires.
    st.cards[0] = ["infantry", "cavalry", "artillery", "infantry"]
    st.cards[1] = ["cavalry"]
    st.cards[2] = []
    st.sets_traded = 0
    st.phase = PH_ATTACK
    st.current_player = 0

    # Resolve attacks until P1's territory is captured (deterministic combat).
    captured = False
    for _ in range(20):
        if st.owner[nb] == 0:
            captured = True
            break
        game.step(st, {"type": "attack", "src": src, "dst": nb})
    if st.owner[nb] == 0:
        captured = True
    assert captured, "P0 should have captured P1's last territory"
    assert st.eliminated[1], "P1 should be eliminated"
    # A forced trade must have fired (P0 had 4 + seized 1 = 5 cards).
    assert st.sets_traded >= 1, "elimination should have triggered a forced trade"
    # Not a conquest win (P2 still holds territory): still in the attack phase.
    assert st.phase == PH_ATTACK
    # The bonus armies were placed on the captured territory, NOT dropped:
    # nb must hold clearly more than the lone defender (1) it started with.
    bonus = set_value(0)  # first set traded
    assert st.armies[nb] >= 1 + bonus, (
        f"forced-trade bonus ({bonus}) was lost; nb has only {st.armies[nb]}")
    # And armies_to_deploy was NOT inflated mid-attack (that pool gets zeroed by
    # _next_player); the bonus lives on the board instead.
    placed = [h for h in st.history if "placed" in h and "forced-trade" in h]
    assert placed, "expected a history line recording the placed forced-trade armies"


# --------------------------------------------------------------------------- #
# Negotiation / messaging lifecycle reachable under RandomPlayer
# --------------------------------------------------------------------------- #

def test_negotiation_messages_and_alliance_events_logged():
    game, result, events = _run(0)
    obs = [e for e in events if e.get("type") == "observation"]
    # Some message observations (say/whisper) should appear.
    msgs = [e for e in obs if e["obs"].get("type") in ("message", "message_meta")]
    assert msgs, "no message observations — negotiation never produced messages"
    # Alliance events appear in the terminal alliance_summary too.
    term = events[-1]
    assert "alliance_summary" in term
    summ = term["alliance_summary"]
    assert summ.get("n_proposed", 0) >= 1, \
        "RandomPlayer should propose at least one alliance over a full match"


def test_alliance_kinds_available():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    actions = game.alliance_legal_actions(st, 0)
    kinds = {a["kind"] for a in actions if a["type"] == "alliance_propose"}
    assert set(ALLIANCE_KINDS) <= kinds


# --------------------------------------------------------------------------- #
# Watcher hooks
# --------------------------------------------------------------------------- #

def test_god_view_and_snapshot_and_board():
    game = RiskLite()
    st = game.initial_state(random.Random(0))
    gv = game.god_view(st)
    assert set(gv) >= {"owner", "armies", "cards", "eliminated"}
    snap = game.snapshot(st)
    assert "public" in snap and "hidden" in snap
    assert "owner" in snap["public"] and "cards" in snap["hidden"]
    board = game.render_board(st, reveal="god")
    assert "Risk board" in board
    # Every continent name shows up in the art.
    for cont in CONTINENTS:
        assert cont in board


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
