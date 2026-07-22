"""Whisper double-count fix for scripts/analyze_game_steering.py.

The runner logs TWO `observation` records of type ``"message"`` for every
whisper: the true recipients' record (``audience`` = sender + recipients)
and a god-log bystander copy (``audience`` = bystanders, produced by
``messaging.py:249-250``'s ``log=dict(full)``). Both carry identical
``(turn, actor, scope, from, to, text)`` content and differ only in
``audience``/``event_id`` -- see the real production-log reproduction in
``docs/results/game_mechanics_audit.md`` ("Whisper double-count" finding).

``objective_metrics()`` (the published `public_msg_ratio`/`msgs_per_match`
numbers) and ``build_transcript()`` (the text fed to the judge) must both
collapse the two records back down to ONE logical whisper.
"""

from __future__ import annotations

import scripts.analyze_game_steering as ags


def _obs(turn, actor, audience, obs):
    o = dict(obs)
    o.setdefault("turn", turn)
    return {"type": "observation", "turn": turn, "actor": actor,
            "audience": audience, "obs": o}


def _fake_match_records():
    """One public `say` (1 record, as the runner actually logs it) + one
    private `whisper` from P1 to P2 (2 records: the true recipients' copy
    audience=[1,2], and the god-log bystander copy audience=[0,3,4]) + a
    terminal record."""
    say = _obs(0, 0, [0, 1, 2, 3, 4],
               {"type": "message", "scope": "public", "from": 0,
                "text": "hello all"})
    whisper_true = _obs(1, 1, [1, 2],
                        {"type": "message", "scope": "private", "from": 1,
                         "to": [2], "text": "secret plan"})
    whisper_godlog = _obs(1, 1, [0, 3, 4],
                          {"type": "message", "scope": "private", "from": 1,
                           "to": [2], "text": "secret plan"})
    terminal = {"type": "terminal", "turn": 2, "rewards": [0.0] * 5,
                "state": {}, "winner": "liberal", "win_reason": "",
                "alliance_summary": {}}
    return [say, whisper_true, whisper_godlog, terminal]


def test_objective_metrics_counts_1_private_and_1_public():
    """Ground truth is 1 say + 1 whisper (2 true messages); the buggy
    pre-fix code counts the whisper twice (pub=1, priv=2, total=3)."""
    recs = _fake_match_records()
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["msgs_per_match"] == 2, obj
    assert obj["public_msg_ratio"] == 0.5, obj


def test_build_transcript_prints_whisper_line_exactly_once():
    recs = _fake_match_records()
    transcript = ags.build_transcript(recs)
    assert transcript.count("secret plan") == 1, transcript
    assert transcript.count("hello all") == 1, transcript


def test_dedup_ignores_audience_and_event_id():
    """The dedup key must not be event_id (always unique, per runner.py's
    fresh-monotonic-id scheme) or audience (differs between the two
    records by construction) -- both records must collapse to one."""
    recs = _fake_match_records()
    # Simulate a real JSONL round-trip: distinct, never-repeating event_ids.
    for i, r in enumerate(recs):
        r["event_id"] = 100 + i
        r["match_id"] = "fake-match"
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["msgs_per_match"] == 2, obj


def test_two_distinct_whispers_both_counted():
    """Sanity check: dedup must not over-collapse two genuinely different
    whispers into one."""
    w1_true = _obs(1, 1, [1, 2],
                   {"type": "message", "scope": "private", "from": 1,
                    "to": [2], "text": "first secret"})
    w1_god = _obs(1, 1, [0, 3, 4],
                  {"type": "message", "scope": "private", "from": 1,
                   "to": [2], "text": "first secret"})
    w2_true = _obs(3, 3, [3, 4],
                   {"type": "message", "scope": "private", "from": 3,
                    "to": [4], "text": "second secret"})
    w2_god = _obs(3, 3, [0, 1, 2],
                  {"type": "message", "scope": "private", "from": 3,
                   "to": [4], "text": "second secret"})
    terminal = {"type": "terminal", "turn": 4, "rewards": [0.0] * 5,
                "state": {}, "winner": "liberal", "win_reason": "",
                "alliance_summary": {}}
    recs = [w1_true, w1_god, w2_true, w2_god, terminal]
    obj = ags.objective_metrics(recs, "secret_hitler")
    assert obj["msgs_per_match"] == 2, obj
    assert obj["public_msg_ratio"] == 0.0, obj
    transcript = ags.build_transcript(recs)
    assert transcript.count("first secret") == 1
    assert transcript.count("second secret") == 1
