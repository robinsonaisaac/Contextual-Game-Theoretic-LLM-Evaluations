"""Task 6: LLM promise-ledger judge ("pass 2") -- kept/broken/renege_rate.

Loads ``scripts/analyze_game_steering.py`` via ``importlib`` (``scripts/`` is
not a package -- no ``__init__.py``), same pattern as
``tests/play/test_transcripts.py``.

Covers:
  * ``judge_promises(client, model_key, transcript)``: valid ledger (2 kept,
    1 broken) parses to a 3-entry list; ``{"promises": []}`` parses to an
    empty list; a garbage (non-JSON) reply returns ``None`` without raising.
  * ``_promise_stats(promises)``: derives ``n_promises``/``n_kept``/
    ``n_broken``/``renege_rate`` from a ledger list (``None`` ledger -> all
    four ``None``; "unresolved" entries count toward ``n_promises`` only).
  * ``--no-promises`` wiring: the promise-judge pass must not be invoked at
    all (asserted via the mock client's call count), while the pass-1
    cooperation/trust/aggression judge still runs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_game_steering",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_game_steering.py",
)
ags = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ags)


class _MockClient:
    """Minimal stand-in for ``LLMClient``: async ``generate`` returning a
    fixed reply text keyed by the requested model, tracking call count."""

    def __init__(self, reply_text):
        self.reply_text = reply_text
        self.calls = []

    async def generate(self, prompt, model=None):
        self.calls.append(prompt)
        return {model: self.reply_text}


VALID_LEDGER = json.dumps({"promises": [
    {"by": "P0", "to": ["P1"], "turn": 1, "promise": "will vote P1 next round",
     "status": "kept", "evidence": "P0 voted P1 at turn 4"},
    {"by": "P1", "to": ["P0"], "turn": 2, "promise": "will share the seer info",
     "status": "kept", "evidence": "P1 revealed the info at turn 3"},
    {"by": "P2", "to": ["P0"], "turn": 3, "promise": "will not accuse P0",
     "status": "broken", "evidence": "P2 accused P0 at turn 6"},
]})

EMPTY_LEDGER = json.dumps({"promises": []})

GARBAGE_REPLY = "Sorry, I will not produce JSON for this request today."


# ------------------------------------------------------------- judge_promises
async def test_judge_promises_valid_ledger_kept_and_broken():
    client = _MockClient(VALID_LEDGER)
    promises = await ags.judge_promises(client, "claude", "some transcript")
    assert promises is not None
    assert len(promises) == 3
    statuses = [p["status"] for p in promises]
    assert statuses.count("kept") == 2
    assert statuses.count("broken") == 1


async def test_judge_promises_empty_ledger_returns_empty_list():
    client = _MockClient(EMPTY_LEDGER)
    promises = await ags.judge_promises(client, "claude", "some transcript")
    assert promises == []


async def test_judge_promises_garbage_reply_returns_none_no_exception():
    client = _MockClient(GARBAGE_REPLY)
    promises = await ags.judge_promises(client, "claude", "some transcript")
    assert promises is None


# --------------------------------------------------------------- _promise_stats
def test_promise_stats_valid_ledger_renege_rate_one_third():
    promises = json.loads(VALID_LEDGER)["promises"]
    stats = ags._promise_stats(promises)
    assert stats["n_promises"] == 3
    assert stats["n_kept"] == 2
    assert stats["n_broken"] == 1
    assert stats["renege_rate"] == 1 / 3


def test_promise_stats_empty_ledger_zeros_renege_none():
    stats = ags._promise_stats([])
    assert stats["n_promises"] == 0
    assert stats["n_kept"] == 0
    assert stats["n_broken"] == 0
    assert stats["renege_rate"] is None


def test_promise_stats_none_ledger_all_four_none():
    stats = ags._promise_stats(None)
    assert stats == {
        "n_promises": None, "n_kept": None, "n_broken": None, "renege_rate": None,
    }


def test_promise_stats_unresolved_counts_toward_total_only():
    promises = [{"status": "kept"}, {"status": "broken"}, {"status": "unresolved"}]
    stats = ags._promise_stats(promises)
    assert stats["n_promises"] == 3
    assert stats["n_kept"] == 1
    assert stats["n_broken"] == 1
    assert stats["renege_rate"] == 0.5


# ------------------------------------------------------------------- wiring
_TERMINAL_LOG = json.dumps({
    "type": "terminal", "turn": 1, "winner": "liberal", "win_reason": "",
    "alliance_summary": {},
}) + "\n"


def _write_match_log(tmp_path):
    log = tmp_path / "match.jsonl"
    log.write_text(_TERMINAL_LOG)
    return {"label": "baseline", "alpha": 0.0, "seed": 0, "log": str(log)}


COOP_REPLY = json.dumps({
    "cooperation_index": 60, "trust_index": 55, "aggression_index": 20,
    "rationale": "steady cooperation",
})


async def test_no_promises_flag_skips_promise_pass(tmp_path):
    """--no-promises must run pass 1 (cooperation/trust/aggression) but never
    invoke the promise-ledger pass -- asserted via the mock client's call
    count (1, not 2)."""
    entry = _write_match_log(tmp_path)
    client = _MockClient(COOP_REPLY)
    sem = asyncio.Semaphore(1)
    row = await ags.process_match(
        entry, client=client, game="secret_hitler", judge_model="claude",
        sem=sem, no_judge=False, no_promises=True,
    )
    assert len(client.calls) == 1
    assert row["n_promises"] is None
    assert row["n_kept"] is None
    assert row["n_broken"] is None
    assert row["renege_rate"] is None
    assert row["promises"] is None
    # pass-1 judge result still present
    assert row["cooperation_index"] == 60.0


async def test_no_judge_flag_skips_both_passes(tmp_path):
    entry = _write_match_log(tmp_path)
    client = _MockClient(COOP_REPLY)
    sem = asyncio.Semaphore(1)
    row = await ags.process_match(
        entry, client=client, game="secret_hitler", judge_model="claude",
        sem=sem, no_judge=True, no_promises=False,
    )
    assert len(client.calls) == 0
    assert row["promises"] is None
    assert "cooperation_index" not in row


async def test_both_passes_run_when_promises_enabled(tmp_path):
    entry = _write_match_log(tmp_path)

    class _TwoReplyClient:
        """Returns the cooperation JSON on the first call, the promise
        ledger JSON on the second -- lets us assert both passes actually ran
        and produced their respective results."""

        def __init__(self):
            self.calls = []

        async def generate(self, prompt, model=None):
            self.calls.append(prompt)
            text = COOP_REPLY if len(self.calls) == 1 else VALID_LEDGER
            return {model: text}

    client = _TwoReplyClient()
    sem = asyncio.Semaphore(1)
    row = await ags.process_match(
        entry, client=client, game="secret_hitler", judge_model="claude",
        sem=sem, no_judge=False, no_promises=False,
    )
    assert len(client.calls) == 2
    assert row["cooperation_index"] == 60.0
    assert row["n_promises"] == 3
    assert row["n_kept"] == 2
    assert row["n_broken"] == 1
    assert row["renege_rate"] == 1 / 3
    assert row["promises"] == json.loads(VALID_LEDGER)["promises"]
