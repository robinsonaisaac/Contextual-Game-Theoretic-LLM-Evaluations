"""Offline metric-reduction tests (metrics.py).

The derived rates are pure log reductions (spec §4). After the per-alliance
redesign (B6/M3/M4), ``_summary_from_events`` and ``load_match`` must dedupe
by ``alliance_id`` exactly like ``alliance_summary`` so the replayed rates are
bounded in [0, 1] and match the live terminal summary, and
``first_betrayal_turn`` must be recoverable from the logged ``alliance_event``
records (m1 / metrics).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from game_theory_llm.play.metrics import (
    _summary_from_events,
    load_match,
)


def _alli_event(event, aid, *, turn, actor=0, members=None, counterparty=None):
    return {
        "type": "alliance_event", "event": event, "alliance_id": aid,
        "turn": turn, "actor": actor,
        "members": members or [], "counterparty": counterparty or [],
        "proposer": (members or [actor])[0],
    }


def test_summary_from_events_dedupes_per_alliance():
    """A single alliance honoured many times then betrayed once is ONE
    betrayed alliance, not many — the buckets stay bounded."""
    events = [
        _alli_event("propose", 0, turn=0, actor=0, members=[0, 1], counterparty=[1]),
        _alli_event("accept", 0, turn=1, actor=1, members=[0, 1], counterparty=[0]),
        _alli_event("honored", 0, turn=2, actor=0, members=[0, 1], counterparty=[1]),
        _alli_event("honored", 0, turn=3, actor=0, members=[0, 1], counterparty=[1]),
        _alli_event("honored", 0, turn=4, actor=0, members=[0, 1], counterparty=[1]),
        _alli_event("betrayed", 0, turn=5, actor=1, members=[0, 1], counterparty=[0]),
    ]
    s = _summary_from_events(events)
    assert s["n_proposed"] == 1
    assert s["n_accepted"] == 1
    assert s["n_honored"] == 0          # betrayal dominates this one alliance
    assert s["n_betrayed"] == 1
    assert s["n_honored"] + s["n_betrayed"] == s["n_accepted"]
    # Raw diagnostics survive.
    assert s["honored_events"] == 3
    assert s["betrayed_events"] == 1


def test_active_alliance_without_betrayal_counts_honored():
    """m1: an active alliance that only ever emits non-betrayal events (or no
    judgement at all once active) is honoured by default, so games that emit
    ONLY 'betrayed' (Risk) still produce a meaningful honour_rate."""
    events = [
        # Alliance 0: active, betrayed.
        _alli_event("propose", 0, turn=0, members=[0, 1], counterparty=[1]),
        _alli_event("accept", 0, turn=1, actor=1, members=[0, 1], counterparty=[0]),
        _alli_event("betrayed", 0, turn=2, actor=0, members=[0, 1], counterparty=[1]),
        # Alliance 1: active, never betrayed (no judgement events at all).
        _alli_event("propose", 1, turn=0, actor=2, members=[2, 3], counterparty=[3]),
        _alli_event("accept", 1, turn=1, actor=3, members=[2, 3], counterparty=[2]),
    ]
    s = _summary_from_events(events)
    assert s["n_accepted"] == 2
    assert s["n_betrayed"] == 1
    assert s["n_honored"] == 1          # alliance 1 honoured by default
    assert s["betrayed_events"] == 1
    assert s["honored_events"] == 0     # no explicit honour event was emitted


def _write_log(records):
    tmp = tempfile.mkdtemp()
    lp = Path(tmp) / "m.jsonl"
    with lp.open("w") as f:
        for i, r in enumerate(records):
            r = dict(r)
            r.setdefault("match_id", "abc")
            r.setdefault("event_id", i)
            f.write(json.dumps(r) + "\n")
    return str(lp)


def test_load_match_rates_bounded_and_first_betrayal():
    """load_match must recompute bounded rates from the logged alliance_event
    records and recover first_betrayal_turn from them."""
    records = [
        {"type": "match_start", "schema": 2, "game": "secret_hitler",
         "n_players": 5, "seed": 1, "config": {}, "steering_tags": [None] * 5},
        _alli_event("propose", 0, turn=0, members=[0, 1], counterparty=[1]),
        _alli_event("accept", 0, turn=1, actor=1, members=[0, 1], counterparty=[0]),
        # Many betrayal occasions on the SAME alliance.
        _alli_event("betrayed", 0, turn=4, actor=1, members=[0, 1], counterparty=[0]),
        _alli_event("betrayed", 0, turn=6, actor=1, members=[0, 1], counterparty=[0]),
        {"type": "terminal", "turn": 7, "rewards": [0.0] * 5, "state": {},
         "winner": "liberal", "win_reason": "", "alliance_summary": {}},
    ]
    # NB: terminal alliance_summary is {} so load_match falls back to recompute.
    lp = _write_log(records)
    m = load_match(lp)
    rates = m["rates"]
    assert rates["formation"] == 1.0
    assert rates["honour"] == 0.0
    assert rates["betrayal"] == 1.0     # one betrayed / one accepted, bounded
    for v in rates.values():
        assert v is None or 0.0 <= v <= 1.0
    # first_betrayal_turn = turn of the first betrayed record.
    assert m["first_betrayal_turn"] == 4


def test_load_match_prefers_terminal_summary():
    """When the terminal record carries a populated alliance_summary, rates
    are derived from it (and remain bounded)."""
    term_summary = {
        "n_proposed": 4, "n_accepted": 3, "n_declined": 1, "n_broken": 1,
        "n_honored": 1, "n_betrayed": 2,
        "honored_events": 9, "betrayed_events": 5, "expired_events": 0,
        "per_player": {},
    }
    records = [
        {"type": "match_start", "schema": 2, "game": "secret_hitler",
         "n_players": 5, "seed": 1, "config": {}, "steering_tags": [None] * 5},
        {"type": "terminal", "turn": 9, "rewards": [0.0] * 5, "state": {},
         "winner": "fascist", "win_reason": "", "alliance_summary": term_summary},
    ]
    lp = _write_log(records)
    m = load_match(lp)
    assert m["alliances"]["n_accepted"] == 3
    assert abs(m["rates"]["formation"] - 3 / 4) < 1e-9
    assert abs(m["rates"]["honour"] - 1 / 3) < 1e-9
    assert abs(m["rates"]["betrayal"] - 2 / 3) < 1e-9
    for v in m["rates"].values():
        assert 0.0 <= v <= 1.0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            if not fn.__name__.startswith("test_"):
                continue
            fn()
    print("ok")
