"""Runner schema-v2 guarantees + end-to-end legacy obs path.

Asserts:
  * every emitted record carries a monotonic ``event_id`` (0,1,2,...) and a
    single shared ``match_id``;
  * ``match_start`` has ``schema == 2`` plus the new ``config`` /
    ``steering_tags`` fields;
  * the new ``setup`` + ``state_snapshot`` records are present;
  * running OneNightWerewolf (which overrides nothing new) end-to-end with
    RandomPlayer still terminates, the ``terminal`` record is last, and the
    legacy broadcast observation path still fires.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from game_theory_llm.play import run_match
from game_theory_llm.play.games import OneNightWerewolf
from game_theory_llm.play.players import RandomPlayer


def _run_onw(seed: int = 7):
    game = OneNightWerewolf()
    players = [RandomPlayer(seed=seed * 10 + i, name=f"R{i}")
               for i in range(game.n_players)]
    tmp = tempfile.mkdtemp()
    log_path = Path(tmp) / "match.jsonl"
    result = run_match(game, players, seed=seed, log_path=log_path)
    events = [json.loads(l) for l in log_path.read_text().splitlines()]
    return game, result, events


def test_event_id_monotonic_from_zero():
    _, _, events = _run_onw()
    ids = [e["event_id"] for e in events]
    assert ids == list(range(len(events))), \
        f"event_id not a 0-based monotonic sequence: {ids[:10]}..."


def test_single_match_id_on_every_record():
    _, _, events = _run_onw()
    match_ids = {e.get("match_id") for e in events}
    assert len(match_ids) == 1
    assert None not in match_ids
    for e in events:
        assert isinstance(e["match_id"], str) and e["match_id"]


def test_match_start_schema_2():
    _, _, events = _run_onw()
    start = events[0]
    assert start["type"] == "match_start"
    assert start["schema"] == 2
    assert "config" in start and isinstance(start["config"], dict)
    assert "steering_tags" in start
    assert len(start["steering_tags"]) == start["n_players"]
    # RandomPlayer has no steering_tag -> all None.
    assert all(t is None for t in start["steering_tags"])


def test_setup_and_snapshot_records_present():
    _, _, events = _run_onw()
    kinds = [e.get("type") for e in events]
    assert "setup" in kinds
    assert "state_snapshot" in kinds
    setup = next(e for e in events if e["type"] == "setup")
    assert "god_view" in setup
    snap = next(e for e in events if e["type"] == "state_snapshot")
    assert "snapshot" in snap and "phase" in snap


def test_onw_terminates_and_terminal_last():
    game, result, events = _run_onw()
    assert game.is_terminal(result.terminal_state)
    assert events[-1]["type"] == "terminal"
    term = events[-1]
    # v2 additive terminal fields present.
    assert "winner" in term
    assert "win_reason" in term
    assert "alliance_summary" in term
    # v2: ONW carries a real alliance ledger, so the terminal summary is the
    # populated reduction dict (the betrayal/honour signal the experiment
    # measures). It is always a dict with the canonical count keys.
    summary = term["alliance_summary"]
    assert isinstance(summary, dict)
    for k in ("n_proposed", "n_accepted", "n_declined", "n_broken",
              "n_honored", "n_betrayed"):
        assert k in summary, f"missing alliance_summary key {k}"


def test_legacy_observation_path_fires():
    """OneNightWerewolf overrides nothing new, so the default broadcast
    observations must still appear (INV-1 end-to-end)."""
    game, _, events = _run_onw()
    obs_records = [e for e in events if e.get("type") == "observation"]
    assert obs_records, "no observation records — legacy obs path regressed"
    # The default broadcast hits all seats with a public 'action' obs.
    bcast = [e for e in obs_records
             if sorted(e["audience"]) == list(range(game.n_players))
             and e["obs"].get("type") == "action"]
    assert bcast, "default broadcast observation missing"


def test_prompt_and_action_carry_phase_and_steering():
    _, _, events = _run_onw()
    prompts = [e for e in events if e.get("type") == "prompt"]
    actions = [e for e in events if e.get("type") == "action"]
    assert prompts and actions
    for e in prompts + actions:
        assert "phase" in e          # additive v2 field
        assert "steering" in e       # additive v2 field (None for RandomPlayer)
        assert e["steering"] is None


def test_phase_change_records_emitted():
    _, _, events = _run_onw()
    pcs = [e for e in events if e.get("type") == "phase_change"]
    # ONW moves night -> day -> vote -> terminal, so at least one phase_change.
    assert pcs, "no phase_change records emitted"
    for pc in pcs:
        assert "from" in pc and "to" in pc and "active_seat" in pc


def test_multiple_seeds_schema_stable():
    for seed in range(5):
        _, result, events = _run_onw(seed)
        ids = [e["event_id"] for e in events]
        assert ids == list(range(len(events)))
        assert events[-1]["type"] == "terminal"


if __name__ == "__main__":
    test_event_id_monotonic_from_zero()
    test_single_match_id_on_every_record()
    test_match_start_schema_2()
    test_setup_and_snapshot_records_present()
    test_onw_terminates_and_terminal_last()
    test_legacy_observation_path_fires()
    test_prompt_and_action_carry_phase_and_steering()
    test_phase_change_records_emitted()
    test_multiple_seeds_schema_stable()
    print("ok")
