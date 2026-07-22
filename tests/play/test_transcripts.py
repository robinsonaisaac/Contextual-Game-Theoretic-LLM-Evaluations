"""Task 5: aborted-match handling, new metrics, game-aware full transcripts.

Loads ``scripts/analyze_game_steering.py`` via ``importlib`` (``scripts/`` is
not a package -- no ``__init__.py``) so this test file is independent of
whatever import machinery pytest/rootdir happens to expose.

Covers:
  * ``objective_metrics(recs, game)``: ``aborted``/``aborted_player``,
    Monopoly-only trade counts (``None`` for other games), ``coop_side_win``
    is ``None`` (not 0.0) for games whose ``COOP_SIDE`` entry is empty/missing.
  * ``build_transcript(recs, game)``: per-game rendering --
    ONW (regression-pinned), Secret Hitler (ja/nein votes, nominate, policy
    enacted), Risk (attack line from the real ``src``/``dst`` action keys),
    Monopoly (event text lines + trade_dialogue offer/accept/reject lines).
  * ``_judge_window``: full transcript under 60k chars passes through
    untouched; over 60k is head(20k) + marker + tail(40k).
  * ``aggregate``: aborted rows excluded from every behaviour-metric mean but
    folded into ``cancel_rate`` (aborted/total), and still counted in
    ``n_matches``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_game_steering",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_game_steering.py",
)
ags = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ags)


def _obs(turn, actor, audience, obs):
    o = dict(obs)
    o.setdefault("turn", turn)
    return {"type": "observation", "turn": turn, "actor": actor,
            "audience": audience, "obs": o}


# --------------------------------------------------------------------- ONW
def test_onw_transcript_regression_pinned():
    """Current ONW rendering (message/alliance/vote/outcome) must stay
    byte-compatible now that build_transcript takes a `game` arg."""
    recs = [
        _obs(0, 0, [0, 1, 2, 3, 4],
             {"type": "message", "scope": "public", "from": 0, "text": "hi all"}),
        {"type": "alliance_event", "alliance_id": 0, "event": "propose",
         "actor": 0, "members": [0, 1]},
        {"type": "action", "turn": 1, "player": 2,
         "action": {"type": "vote", "target": 4}},
        {"type": "terminal", "turn": 2, "winner": "village",
         "win_reason": "wolves eliminated", "alliance_summary": {}},
    ]
    t = ags.build_transcript(recs, "one_night_werewolf")
    assert "P0 (public): hi all" in t
    assert "ALLIANCE #0 propose by P0 (members [0, 1])" in t
    assert "P2 VOTES P4" in t
    assert "OUTCOME winner=village (wolves eliminated)" in t


# ---------------------------------------------------------------- Secret Hitler
def test_sh_transcript_votes_nominate_enact():
    recs = [
        {"type": "action", "turn": 0, "player": 1,
         "action": {"type": "nominate", "target": 3}},
        {"type": "action", "turn": 1, "player": 0,
         "action": {"type": "vote", "ja": True}},
        {"type": "action", "turn": 1, "player": 1,
         "action": {"type": "vote", "ja": False}},
        {"type": "action", "turn": 2, "player": 3,
         "action": {"type": "enact", "index": 0}},
        {"type": "state_snapshot", "turn": 2,
         "snapshot": {"public": {"enacted_liberal": 0, "enacted_fascist": 1}}},
        {"type": "terminal", "turn": 3, "winner": "fascist",
         "win_reason": "6 fascist policies", "alliance_summary": {}},
    ]
    t = ags.build_transcript(recs, "secret_hitler")
    assert "P1 nominates P3 as chancellor" in t
    assert "P0 votes ja" in t
    assert "P1 votes nein" in t
    assert "VOTES PNone" not in t
    assert "policy enacted: fascist" in t


def test_sh_transcript_veto_lines():
    recs = [
        {"type": "action", "turn": 5, "player": 1, "action": {"type": "veto"}},
        {"type": "action", "turn": 5, "player": 2,
         "action": {"type": "veto_consent", "agree": True}},
    ]
    t = ags.build_transcript(recs, "secret_hitler")
    assert "P1" in t and "veto" in t.lower()
    assert "P2" in t


# -------------------------------------------------------------------- Risk
def test_risk_transcript_attack_line():
    armies = [0] * 42
    armies[34] = 6
    recs = [
        {"type": "state_snapshot", "turn": 5,
         "snapshot": {"public": {"armies": armies}}},
        {"type": "action", "turn": 6, "player": 0,
         "action": {"type": "attack", "src": 34, "dst": 27}},
        {"type": "observation", "turn": 6, "actor": 0,
         "obs": {"type": "combat", "src": 34, "dst": 27,
                 "atk_rolls": [5, 4, 3], "def_rolls": [2],
                 "atk_lost": 0, "def_lost": 1, "captured": True, "actor": 0}},
        {"type": "terminal", "turn": 7, "winner": 0,
         "win_reason": "controls all 42 territories", "alliance_summary": {}},
    ]
    t = ags.build_transcript(recs, "risk")
    assert "P0 attacks 34->27" in t
    assert "6 armies" in t
    assert "captur" in t.lower()
    assert "OUTCOME winner=0" in t


# --------------------------------------------------------------- Monopoly
def test_monopoly_transcript_events_and_trades():
    recs = [
        {"type": "observation", "turn": 3,
         "obs": {"type": "event", "text": "P0 bought purple1 for $60"}},
        {"type": "observation", "turn": 4,
         "obs": {"type": "trade_dialogue", "event": "propose_trade",
                 "trade": {"from": "P0", "to": "P1",
                           "give_props": ["purple1"], "give_cash": 50,
                           "want_props": ["purple2"], "want_cash": 0,
                           "message": "deal?"},
                 "message": "deal?"}},
        {"type": "observation", "turn": 5,
         "obs": {"type": "trade_dialogue", "event": "accept_trade",
                 "trade": {"from": "P0", "to": "P1",
                           "give_props": ["purple1"], "give_cash": 50,
                           "want_props": ["purple2"], "want_cash": 0,
                           "message": "deal?"},
                 "message": "sure"}},
        {"type": "terminal", "turn": 200, "winner": "P0",
         "win_reason": "turn cap net worth", "alliance_summary": {}},
    ]
    t = ags.build_transcript(recs, "monopoly_lite")
    assert "P0 bought purple1 for $60" in t
    assert "P0 -> P1 OFFER" in t
    assert "P1 ACCEPTS" in t
    assert "OUTCOME winner=P0" in t


# --------------------------------------------------------------- truncation
def test_judge_window_full_transcript_passes_through():
    t = "short transcript OUTCOME winner=village ()"
    assert ags._judge_window(t) == t


def test_judge_window_head_tail_marker_and_outcome_survives():
    body = "x" * 70000
    t = "HEAD_START" + body + "OUTCOME winner=village (TAIL_END)"
    n = len(t) - 60000
    w = ags._judge_window(t)
    assert len(w) < len(t)
    assert w.startswith(t[:20000])
    assert w.endswith(t[-40000:])
    assert f"[... {n} chars omitted ...]" in w
    assert "OUTCOME" in w


# ----------------------------------------------------------------- aborted
def test_aborted_objective_metrics():
    recs = [
        {"type": "match_start", "schema": 2, "game": "secret_hitler"},
        {"type": "aborted", "reason": "unparseable_output", "turn": 5,
         "player": 2, "model": "gemma", "steering": None,
         "phase": "discussion", "last_error": "bad", "last_raw": "garbage"},
    ]
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["aborted"] is True
    assert obj["aborted_player"] == 2
    assert obj["coop_side_win"] is None
    assert obj["winner"] is None


def test_non_aborted_has_aborted_false():
    recs = [{"type": "terminal", "turn": 1, "winner": "liberal",
             "win_reason": "", "alliance_summary": {}}]
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["aborted"] is False
    assert obj["aborted_player"] is None


def test_legacy_fallback_still_counted():
    recs = [
        {"type": "fallback", "turn": 3, "player": 0, "action": {}, "reason": "x"},
        {"type": "terminal", "turn": 4, "winner": "liberal",
         "win_reason": "", "alliance_summary": {}},
    ]
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["n_fallback"] == 1
    assert obj["aborted"] is False


# ------------------------------------------------------------- game aliases
def test_risk_lite_jobs_json_alias_dispatches_to_risk_builder():
    """`jobs.json`'s `game` field for Risk runs is the registry key
    "risk_lite", not the game class's own `.name` ("risk", also the JSONL
    logs' top-level `game` field) -- both objective_metrics and
    build_transcript must treat them identically."""
    armies = [0] * 42
    armies[34] = 6
    recs = [
        {"type": "state_snapshot", "turn": 5,
         "snapshot": {"public": {"armies": armies}}},
        {"type": "action", "turn": 6, "player": 0,
         "action": {"type": "attack", "src": 34, "dst": 27}},
        {"type": "terminal", "turn": 7, "winner": 0,
         "win_reason": "controls all 42 territories", "alliance_summary": {}},
    ]
    t_alias = ags.build_transcript(recs, "risk_lite")
    t_canon = ags.build_transcript(recs, "risk")
    assert t_alias == t_canon
    assert "P0 attacks 34->27 with 6 armies" in t_alias

    obj_alias = ags.objective_metrics(recs, "risk_lite")
    obj_canon = ags.objective_metrics(recs, "risk")
    assert obj_alias == obj_canon
    assert obj_alias["coop_side_win"] is None


# ------------------------------------------------------------ coop_side_win
def test_coop_side_win_none_for_non_coop_games():
    term = {"type": "terminal", "turn": 1, "winner": "P0", "win_reason": "",
            "alliance_summary": {}}
    assert ags.objective_metrics([term], "monopoly_lite")["coop_side_win"] is None
    assert ags.objective_metrics([term], "risk")["coop_side_win"] is None


def test_coop_side_win_still_numeric_for_onw_and_sh():
    term_village = {"type": "terminal", "turn": 1, "winner": "village",
                     "win_reason": "", "alliance_summary": {}}
    assert ags.objective_metrics(
        [term_village], "one_night_werewolf")["coop_side_win"] == 1.0
    term_fascist = {"type": "terminal", "turn": 1, "winner": "fascist",
                     "win_reason": "", "alliance_summary": {}}
    assert ags.objective_metrics(
        [term_fascist], "secret_hitler")["coop_side_win"] == 0.0


# ---------------------------------------------------------------- trades
def test_monopoly_trade_counts_and_none_elsewhere():
    recs = [
        {"type": "observation",
         "obs": {"type": "trade_dialogue", "event": "propose_trade", "trade": {}}},
        {"type": "observation",
         "obs": {"type": "trade_dialogue", "event": "propose_trade", "trade": {}}},
        {"type": "observation",
         "obs": {"type": "trade_dialogue", "event": "accept_trade", "trade": {}}},
        {"type": "observation",
         "obs": {"type": "trade_dialogue", "event": "reject_trade", "trade": {}}},
        {"type": "terminal", "turn": 5, "winner": "P0", "win_reason": "",
         "alliance_summary": {}},
    ]
    obj = ags.objective_metrics(recs, "monopoly_lite")
    assert obj["n_trades_proposed"] == 2
    assert obj["n_trades_completed"] == 1
    assert obj["n_trades_rejected"] == 1

    obj2 = ags.objective_metrics(recs, "secret_hitler")
    assert obj2["n_trades_proposed"] is None
    assert obj2["n_trades_completed"] is None
    assert obj2["n_trades_rejected"] is None


# ------------------------------------------------------------- aggregation
def test_cancel_rate_aggregate_one_of_two():
    aborted_row = {"label": "baseline", "alpha": 0.0, "seed": 0,
                   **ags.objective_metrics(
                       [{"type": "aborted", "reason": "unparseable_output",
                         "turn": 1, "player": 0}], "secret_hitler")}
    ok_row = {"label": "baseline", "alpha": 0.0, "seed": 1,
              **ags.objective_metrics(
                  [{"type": "terminal", "turn": 10, "winner": "liberal",
                    "win_reason": "", "alliance_summary": {}}], "secret_hitler")}
    agg = ags.aggregate([aborted_row, ok_row])
    assert agg["baseline"]["n_matches"] == 2
    assert agg["baseline"]["cancel_rate"]["mean"] == 0.5
    # the aborted row must be excluded from every behaviour-metric mean --
    # only the "liberal" (coop-side) winner row contributes, so the mean is
    # exactly 1.0, not diluted by the aborted row.
    assert agg["baseline"]["coop_side_win"]["n"] == 1
    assert agg["baseline"]["coop_side_win"]["mean"] == 1.0
