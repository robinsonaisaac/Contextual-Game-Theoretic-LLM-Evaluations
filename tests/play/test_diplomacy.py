"""Tests for the standard-map Full-Press Diplomacy game (Unit 4).

Three layers:

1. A DATC-style subset exercised against the PURE ``adjudicate(orders, board)``
   function: support, support-cut, standoff bounce, single-fleet convoy,
   self-dislodgement ban, dislodgement, retreat-or-disband, build/disband
   counts. Out-of-subset DATC cases are explicitly ``pytest.mark.skip`` with a
   reason (multi-fleet convoy chains, convoy paradox fixpoints, split coasts).

2. A full 7-player ``RandomPlayer`` match that must terminate within
   ``max_turns=200`` with a valid winner.

3. Full-Press alliance instrumentation: one support-pact honoured scenario and
   one betrayed scenario driven through ``_judge_movement``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from game_theory_llm.play import GameConfig, run_match
from game_theory_llm.play.players import RandomPlayer
from game_theory_llm.play.maps.diplomacy_map import pid as P
from game_theory_llm.play.games.diplomacy_lite import (
    DiplomacyLite,
    Board,
    adjudicate,
    POWERS,
    PH_RETREAT,
    PH_WINTER,
    MAX_YEARS,
)
from game_theory_llm.play.alliances import Alliance


# ===========================================================================
# 1. DATC-style subset on the pure adjudicator.
# ===========================================================================
def _move(dst, convoy=False):
    return {"type": "MOVE", "target": dst, "via_convoy": convoy}


def _sup_move(atk, dst):
    return {"type": "SUPPORT", "attacker": atk, "target": dst}


def _sup_hold(tgt):
    return {"type": "SUPPORT", "target": tgt}


def test_datc_simple_hold_no_dislodge():
    """A lone holding unit with no attacker is untouched."""
    b = Board(units={P("PAR"): ("A", "France")})
    r = adjudicate({P("PAR"): {"type": "HOLD"}}, b)
    assert r.moves == {}
    assert P("PAR") in r.holds
    assert r.dislodged == {}


def test_datc_move_to_empty_succeeds():
    b = Board(units={P("PAR"): ("A", "France")})
    r = adjudicate({P("PAR"): _move(P("BUR"))}, b)
    assert r.moves == {P("PAR"): P("BUR")}


def test_datc_standoff_bounce():
    """Two equal-strength armies moving to the same empty province bounce."""
    b = Board(units={P("PAR"): ("A", "France"), P("MUN"): ("A", "Germany")})
    r = adjudicate({P("PAR"): _move(P("BUR")), P("MUN"): _move(P("BUR"))}, b)
    assert r.moves == {}
    assert P("BUR") in r.bounces


def test_datc_support_breaks_standoff():
    b = Board(units={P("PAR"): ("A", "France"), P("MAR"): ("A", "France"),
                     P("MUN"): ("A", "Germany")})
    r = adjudicate({
        P("PAR"): _move(P("BUR")),
        P("MAR"): _sup_move(P("PAR"), P("BUR")),
        P("MUN"): _move(P("BUR")),
    }, b)
    assert r.moves.get(P("PAR")) == P("BUR")
    assert P("MUN") not in r.moves


def test_datc_support_cut_by_other_attacker():
    """DATC 6.D.* family: an attack on the supporter cuts the support."""
    b = Board(units={
        P("PAR"): ("A", "France"), P("MAR"): ("A", "France"),
        P("MUN"): ("A", "Germany"), P("PIE"): ("A", "Italy"),
    })
    r = adjudicate({
        P("PAR"): _move(P("BUR")),
        P("MAR"): _sup_move(P("PAR"), P("BUR")),   # support cut by PIE->MAR
        P("MUN"): _move(P("BUR")),
        P("PIE"): _move(P("MAR")),
    }, b)
    assert P("MAR") in r.cut_supports
    # PAR(1) vs MUN(1) standoff -> BUR bounces, nobody moves in.
    assert P("BUR") in r.bounces
    assert r.moves.get(P("PAR")) != P("BUR")


def test_datc_dislodgement_with_support():
    """A 2-strength supported attack dislodges a 1-strength holder."""
    b = Board(units={P("PAR"): ("A", "France"), P("MAR"): ("A", "France"),
                     P("BUR"): ("A", "Germany")})
    r = adjudicate({
        P("PAR"): _move(P("BUR")),
        P("MAR"): _sup_move(P("PAR"), P("BUR")),
        P("BUR"): {"type": "HOLD"},
    }, b)
    assert r.moves.get(P("PAR")) == P("BUR")
    assert P("BUR") in r.dislodged
    assert r.dislodged[P("BUR")]["power"] == "Germany"
    assert r.dislodged[P("BUR")]["from"] == P("PAR")


def test_datc_self_dislodgement_ban():
    """A power may never dislodge its own unit, even with support."""
    b = Board(units={P("PAR"): ("A", "France"), P("MAR"): ("A", "France"),
                     P("BUR"): ("A", "France")})
    r = adjudicate({
        P("PAR"): _move(P("BUR")),
        P("MAR"): _sup_move(P("PAR"), P("BUR")),
        P("BUR"): {"type": "HOLD"},
    }, b)
    assert r.moves == {}
    assert P("BUR") not in r.dislodged
    assert P("BUR") in r.bounces


def test_datc_single_fleet_convoy_succeeds():
    """A LON - BEL convoyed by a single fleet F NTH succeeds."""
    b = Board(units={P("LON"): ("A", "England"), P("NTH"): ("F", "England")})
    r = adjudicate({
        P("LON"): _move(P("BEL"), convoy=True),
        P("NTH"): {"type": "CONVOY", "army": P("LON"), "target": P("BEL")},
    }, b)
    assert r.moves.get(P("LON")) == P("BEL")


def test_datc_convoy_without_fleet_fails():
    """A via-convoy move with no matching convoying fleet fails (holds)."""
    b = Board(units={P("LON"): ("A", "England")})
    r = adjudicate({P("LON"): _move(P("BEL"), convoy=True)}, b)
    assert r.moves == {}
    assert P("LON") in r.failed_convoys


def test_datc_support_hold_resists_attack():
    """A supported hold (strength 2) resists an unsupported attack (1)."""
    b = Board(units={P("BUR"): ("A", "France"), P("MAR"): ("A", "France"),
                     P("MUN"): ("A", "Germany")})
    r = adjudicate({
        P("BUR"): {"type": "HOLD"},
        P("MAR"): _sup_hold(P("BUR")),
        P("MUN"): _move(P("BUR")),
    }, b)
    assert P("BUR") not in r.dislodged
    assert r.moves == {}


def test_datc_army_cannot_move_unsupported_into_equal_defender():
    """A 1-strength attack cannot dislodge a 1-strength holder."""
    b = Board(units={P("PAR"): ("A", "France"), P("BUR"): ("A", "Germany")})
    r = adjudicate({P("PAR"): _move(P("BUR")), P("BUR"): {"type": "HOLD"}}, b)
    assert r.moves == {}
    assert P("BUR") not in r.dislodged


# ---- Out-of-subset DATC cases: explicitly skipped with a reason. -----------
@pytest.mark.skip(reason="multi-fleet convoy chains disallowed (lite-press-v2 §9): "
                         "ParseError at parse time, never adjudicated")
def test_datc_multi_fleet_convoy_chain():
    pass


@pytest.mark.skip(reason="convoy paradoxes resolved by Szykman rule (failed/hold); "
                         "no fixpoint solver in lite-press-v2 §9")
def test_datc_convoy_paradox_pandins():
    pass


@pytest.mark.skip(reason="split coasts modelled as single provinces (lite-press-v2 §9); "
                         "no north/south-coast adjudication")
def test_datc_split_coast_disambiguation():
    pass


@pytest.mark.skip(reason="beleaguered-garrison nuance resolves by deterministic "
                         "strength comparison; not separately special-cased (§9)")
def test_datc_beleaguered_garrison():
    pass


# ===========================================================================
# 2. Retreat-or-disband and build/disband counts (via the game).
# ===========================================================================
def _fresh_game():
    g = DiplomacyLite()
    st = g.initial_state(__import__("random").Random(0))
    return g, st


def test_retreat_or_disband_auto_target():
    """A dislodged unit retreats to the lowest-index legal empty space, else
    disbands. We check the deterministic auto-retreat target helper."""
    g, st = _fresh_game()
    # Manufacture a dislodgement: a German army dislodged from MUN, attacker
    # came from RUH. Legal retreats exclude RUH, occupied provinces, contested.
    st.units = {P("MUN"): ("A", "Germany")}
    st.dislodged = {P("MUN"): {"power": "Germany", "type": "A", "from": P("RUH")}}
    st.contested = set()
    tgt = g._auto_retreat_target(st, P("MUN"))
    assert tgt is not None
    assert tgt != P("RUH")             # cannot retreat whence the attacker came
    assert tgt not in st.units         # must be empty
    # Now block every legal target -> must disband (None).
    g2, st2 = _fresh_game()
    st2.units = {P("MUN"): ("A", "Germany")}
    # Fill all army-neighbours of MUN with units so none are empty.
    from game_theory_llm.play.maps.diplomacy_map import ADJ_ARMY
    for nb in ADJ_ARMY[P("MUN")]:
        st2.units[nb] = ("A", "Austria")
    st2.dislodged = {P("MUN"): {"power": "Germany", "type": "A", "from": P("RUH")}}
    st2.contested = set()
    assert g2._auto_retreat_target(st2, P("MUN")) is None


def test_build_count_from_sc_surplus():
    """A power with more SCs than units may build the difference on empty home
    centres (excess waived)."""
    g, st = _fresh_game()
    # Give France 4 SCs (3 home + POR) but only 2 units -> build delta +2.
    france_home = {P("BRE"), P("PAR"), P("MAR")}
    st.units = {P("PAR"): ("A", "France"), P("MAR"): ("A", "France"),
                P("MUN"): ("A", "Germany")}
    st.sc_owner = {pid: "France" for pid in france_home}
    st.sc_owner[P("POR")] = "France"
    assert g._build_delta(st, "France") == 2
    sites = g._buildable_sites(st, "France")
    # Only BRE is an empty home SC (PAR/MAR are occupied).
    assert P("BRE") in sites
    assert P("PAR") not in sites and P("MAR") not in sites
    # Apply builds: BUILD on BRE; the second build is waived (no empty home SC).
    g._apply_builds(st, g.seat_of("France"),
                    {P("BRE"): {"type": "BUILD", "unit": "F"}})
    assert st.units.get(P("BRE")) == ("F", "France")
    # France still has only 3 units (one excess build waived).
    assert len(g._units_of(st, "France")) == 3


def test_disband_count_from_sc_deficit():
    """A power with fewer SCs than units must disband the difference,
    lowest-province-index first when unspecified."""
    g, st = _fresh_game()
    # Germany owns 1 SC but 3 units -> must disband 2.
    st.units = {P("KIE"): ("F", "Germany"), P("BER"): ("A", "Germany"),
                P("MUN"): ("A", "Germany")}
    st.sc_owner = {P("KIE"): "Germany"}
    assert g._build_delta(st, "Germany") == -2
    g._apply_builds(st, g.seat_of("Germany"), {})   # auto-disband
    remaining = g._units_of(st, "Germany")
    assert len(remaining) == 1
    # Lowest-index units disbanded first: KIE(37) and BER(40) gone, MUN(39)?
    # Province ids: BER=40, KIE=37, MUN=39. Lowest two = KIE(37), MUN(39).
    # So the survivor is BER.
    assert remaining == [P("BER")]


# ===========================================================================
# 3. Full 7-player RandomPlayer match terminates with a valid winner.
# ===========================================================================
def test_random_match_terminates_with_winner():
    for seed in range(4):
        game = DiplomacyLite()
        players = [RandomPlayer(seed=seed * 17 + i)
                   for i in range(game.n_players)]
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / f"dip_{seed}.jsonl"
            result = run_match(game, players, seed=seed, log_path=log_path,
                               max_turns=200)
            assert game.is_terminal(result.terminal_state), \
                f"seed {seed} did not terminate"
            w = result.terminal_state.winner
            assert w is not None and 0 <= w < game.n_players
            assert sum(result.rewards) == 1.0
            assert result.rewards[w] == 1.0
            events = [json.loads(l) for l in log_path.read_text().splitlines()]
            kinds = {e.get("type") for e in events}
            assert "match_start" in kinds
            assert "terminal" in kinds
            assert any(e.get("type") == "prompt" for e in events)
            assert any(e.get("type") == "action" for e in events)
            assert events[-1]["type"] == "terminal"
            # schema v2 stamping.
            assert events[0]["schema"] == 2
            assert all("event_id" in e and "match_id" in e for e in events)


def test_year_cap_bounds_match_length():
    """The hard year cap guarantees the match never runs forever."""
    game = DiplomacyLite()
    players = [RandomPlayer(seed=100 + i) for i in range(game.n_players)]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cap.jsonl"
        result = run_match(game, players, seed=99, log_path=log_path,
                           max_turns=200)
        assert game.is_terminal(result.terminal_state)
        # Year never exceeds the cap by more than the final +1 increment.
        assert result.terminal_state.year <= 1900 + MAX_YEARS + 1


# ===========================================================================
# 4. Full-Press: support-pact honoured + betrayed scenarios.
# ===========================================================================
def _activate_alliance(state, members, kind, *, accepted_turn=1):
    """Inject an active alliance directly into the alliance state."""
    al = Alliance(
        id=state.alli.next_id,
        members=sorted(members),
        proposer=members[0],
        kind=kind,
        terms={"text": "I will support you"},
        status="active",
        proposed_turn=0,
        accepted_turn=accepted_turn,
        pending=[],
    )
    state.alli.alliances[al.id] = al
    state.alli.next_id += 1
    return al


def test_support_pact_honored():
    """A support_pact is HONORED when a power's orders include a support for an
    allied unit."""
    g, st = _fresh_game()
    st.turn = 5
    # France (seat 2) and Germany (seat 3) form a support_pact.
    fseat, gseat = g.seat_of("France"), g.seat_of("Germany")
    al = _activate_alliance(st, [fseat, gseat], "support_pact")
    # Board: France A BUR supports German A MUN -> RUH (an ally's move).
    st.units = {P("BUR"): ("A", "France"), P("MUN"): ("A", "Germany")}
    orders = {
        P("BUR"): _sup_move(P("MUN"), P("RUH")),   # France supports Germany
        P("MUN"): _move(P("RUH")),                  # German move
    }
    from game_theory_llm.play.games.diplomacy_lite import adjudicate as adj, Board
    res = adj(orders, Board(units=dict(st.units)))
    g._judge_movement(st, orders, res)
    honored = [e for e in st.alli.events if e["event"] == "honored"]
    assert honored, "expected at least one honored event"
    # France is the honourer.
    assert any(e["actor"] == fseat for e in honored)


def test_support_pact_betrayed_by_move_into_ally():
    """An active alliance is BETRAYED when a member MOVEs into an ally's
    occupied province."""
    g, st = _fresh_game()
    st.turn = 6
    fseat, gseat = g.seat_of("France"), g.seat_of("Germany")
    _activate_alliance(st, [fseat, gseat], "nonaggression")
    # France A BUR -> MUN, where Germany occupies MUN -> betrayal of Germany.
    st.units = {P("BUR"): ("A", "France"), P("MUN"): ("A", "Germany")}
    orders = {P("BUR"): _move(P("MUN")), P("MUN"): {"type": "HOLD"}}
    from game_theory_llm.play.games.diplomacy_lite import adjudicate as adj, Board
    res = adj(orders, Board(units=dict(st.units)))
    g._judge_movement(st, orders, res)
    betrayed = [e for e in st.alli.events if e["event"] == "betrayed"]
    assert betrayed, "expected a betrayed event"
    ev = betrayed[0]
    assert ev["actor"] == fseat
    assert gseat in ev["counterparty"]


def test_support_pact_betrayed_by_omission():
    """A support_pact with NO supporting order present is an omission
    betrayal."""
    g, st = _fresh_game()
    st.turn = 7
    fseat, gseat = g.seat_of("France"), g.seat_of("Germany")
    _activate_alliance(st, [fseat, gseat], "support_pact")
    # France just holds and moves; never supports Germany -> omission betrayal.
    st.units = {P("BUR"): ("A", "France"), P("MUN"): ("A", "Germany")}
    orders = {P("BUR"): {"type": "HOLD"}, P("MUN"): {"type": "HOLD"}}
    from game_theory_llm.play.games.diplomacy_lite import adjudicate as adj, Board
    res = adj(orders, Board(units=dict(st.units)))
    g._judge_movement(st, orders, res)
    betrayed = [e for e in st.alli.events
                if e["event"] == "betrayed" and e["actor"] == fseat]
    assert betrayed, "expected an omission betrayal by France"
    assert betrayed[0]["action_ref"]["reason"] == "omitted_promised_support"


def test_alliance_summary_counts_honor_and_betray():
    """The terminal alliance_summary reduction reflects emitted events."""
    from game_theory_llm.play.alliances import alliance_summary
    g, st = _fresh_game()
    st.turn = 3
    fseat, gseat = g.seat_of("France"), g.seat_of("Germany")
    _activate_alliance(st, [fseat, gseat], "support_pact")
    st.units = {P("BUR"): ("A", "France"), P("MUN"): ("A", "Germany")}
    # Honour: France supports Germany.
    orders = {P("BUR"): _sup_move(P("MUN"), P("RUH")), P("MUN"): _move(P("RUH"))}
    from game_theory_llm.play.games.diplomacy_lite import adjudicate as adj, Board
    res = adj(orders, Board(units=dict(st.units)))
    g._judge_movement(st, orders, res)
    summ = alliance_summary(st.alli)
    assert summ["n_honored"] >= 1
    assert summ["per_player"][str(fseat)]["honored"] >= 1


if __name__ == "__main__":
    import sys
    import pytest as _pt
    sys.exit(_pt.main([__file__, "-v"]))
