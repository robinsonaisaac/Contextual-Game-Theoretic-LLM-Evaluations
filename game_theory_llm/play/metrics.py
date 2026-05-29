"""Offline steering-metric reductions over JSONL match logs.

Everything here is a *pure log reduction* (spec §4): the harness exists to
measure steering, so the metrics are recovered from the log alone — no game
engine needed. Message sentiment / aggression / deception are scored
post-hoc by an LLM judge (Sonnet+), NEVER by regex (project rule); that path
lives in ``score_messages`` and is a documented stub until a judge client is
wired in.

Derived metrics (per spec §4):
  * formation rate   = n_accepted / n_proposed
  * betrayal rate    = n_betrayed / n_accepted
  * honour rate      = n_honored / n_accepted
  * private-message ratio = count(message scope=private) / count(message)
  * first-strike turn = turn of the first ``betrayed`` event
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------- loading
def _read_events(log_path: str) -> List[dict]:
    """Parse a JSONL log, tolerating a partial trailing line (for --follow)."""
    events: List[dict] = []
    path = Path(log_path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip an un-terminated trailing line.
                continue
    return events


def load_match(log_path: str) -> dict:
    """Load one match log and reduce it to a metrics-ready summary dict.

    Returns
    -------
    dict with keys:
      match_id, game, n_players, seed, schema, config, steering_tags,
      events (raw list), n_events,
      messages: {total, public, private, meta, private_ratio},
      alliances: the terminal ``alliance_summary`` (or recomputed from
                 ``alliance_event`` records if absent),
      rates: {formation, betrayal, honour},
      first_betrayal_turn, winner, win_reason, rewards.
    """
    events = _read_events(log_path)
    start = next((e for e in events if e.get("type") == "match_start"), {})
    terminal = next((e for e in reversed(events)
                     if e.get("type") == "terminal"), {})

    # Message tallies from observation records (use the logged god-view obs so
    # private messages are counted exactly once at their full content).
    n_public = n_private = n_meta = 0
    seen_msgs = set()
    for e in events:
        if e.get("type") != "observation":
            continue
        obs = e.get("obs", {})
        ot = obs.get("type")
        key = (e.get("event_id"),)
        if ot == "message":
            if obs.get("scope") == "private":
                if key not in seen_msgs:
                    n_private += 1
                    seen_msgs.add(key)
            else:
                if key not in seen_msgs:
                    n_public += 1
                    seen_msgs.add(key)
        elif ot == "message_meta":
            n_meta += 1
    total_msgs = n_public + n_private

    # Alliance summary: prefer the terminal record, else recompute from events.
    alli = terminal.get("alliance_summary") or _summary_from_events(events)
    n_proposed = alli.get("n_proposed", 0)
    n_accepted = alli.get("n_accepted", 0)
    n_betrayed = alli.get("n_betrayed", 0)
    n_honored = alli.get("n_honored", 0)

    def _ratio(num: int, den: int) -> Optional[float]:
        return (num / den) if den else None

    first_betrayal_turn = None
    for e in events:
        if e.get("type") == "alliance_event" and e.get("event") == "betrayed":
            first_betrayal_turn = e.get("turn")
            break

    return {
        "match_id": start.get("match_id"),
        "game": start.get("game"),
        "n_players": start.get("n_players"),
        "seed": start.get("seed"),
        "schema": start.get("schema"),
        "config": start.get("config", {}),
        "steering_tags": start.get("steering_tags", []),
        "events": events,
        "n_events": len(events),
        "messages": {
            "total": total_msgs, "public": n_public, "private": n_private,
            "meta": n_meta,
            "private_ratio": _ratio(n_private, total_msgs),
        },
        "alliances": alli,
        "rates": {
            "formation": _ratio(n_accepted, n_proposed),
            "betrayal": _ratio(n_betrayed, n_accepted),
            "honour": _ratio(n_honored, n_accepted),
        },
        "first_betrayal_turn": first_betrayal_turn,
        "winner": terminal.get("winner"),
        "win_reason": terminal.get("win_reason", ""),
        "rewards": terminal.get("rewards", []),
    }


def _summary_from_events(events: List[dict]) -> dict:
    """Recompute the §4 alliance summary purely from ``alliance_event``
    records (used when a log predates the terminal-record summary)."""
    counts = Counter()
    for e in events:
        if e.get("type") == "alliance_event":
            counts[e.get("event")] += 1
    return {
        "n_proposed": counts.get("propose", 0),
        "n_accepted": counts.get("accept", 0),
        "n_declined": counts.get("decline", 0),
        "n_broken": counts.get("break", 0),
        "n_honored": counts.get("honored", 0),
        "n_betrayed": counts.get("betrayed", 0),
        "per_player": {},
    }


# ----------------------------------------------------------------- table
def steering_table(logs: List[str]) -> "Any":
    """Build a pandas DataFrame of one row per match, ready to group by the
    steering arm. The steering arm is derived from each match's
    ``steering_tags`` (the distinct non-null tag set, joined).
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for lp in logs:
        m = load_match(lp)
        tags = [t for t in (m.get("steering_tags") or []) if t]
        arm = _arm_label(tags)
        msgs = m["messages"]
        rates = m["rates"]
        rows.append({
            "match_id": m.get("match_id"),
            "game": m.get("game"),
            "seed": m.get("seed"),
            "arm": arm,
            "n_proposed": m["alliances"].get("n_proposed", 0),
            "n_accepted": m["alliances"].get("n_accepted", 0),
            "n_betrayed": m["alliances"].get("n_betrayed", 0),
            "n_honored": m["alliances"].get("n_honored", 0),
            "formation_rate": rates.get("formation"),
            "betrayal_rate": rates.get("betrayal"),
            "honour_rate": rates.get("honour"),
            "n_messages": msgs.get("total", 0),
            "private_msg_ratio": msgs.get("private_ratio"),
            "first_betrayal_turn": m.get("first_betrayal_turn"),
            "winner": m.get("winner"),
        })
    return pd.DataFrame(rows)


def _arm_label(tags: List[Any]) -> str:
    """Collapse a match's steering tags into a single arm label."""
    if not tags:
        return "baseline"
    labels = []
    for t in tags:
        if isinstance(t, dict):
            vec = t.get("vector")
            alpha = t.get("alpha")
            labels.append(f"{vec}@{alpha}")
        else:
            labels.append(str(t))
    return "|".join(sorted(set(labels)))


# ----------------------------------------------------------------- scoring
def score_messages(log_path: str, judge_client) -> None:
    """Score each message in a log with an LLM judge (Sonnet+) and append
    ``msg_score`` enrichment records keyed by the annotated ``event_id``.

    Per project rule, message quality (sentiment / aggression / deception) is
    judged by an LLM, NEVER by regex. This is a documented stub: it raises
    ``NotImplementedError`` unless a judge client is supplied, so callers must
    wire in a real judge before using it. The expected output record shape
    (spec §4) is::

        {"type": "msg_score", "event_id": int, "scored_by": str,
         "sentiment": float, "aggression": float,
         "deception_flag": bool, "rationale": str}

    Implementation contract for a future judge client: it must expose a
    blocking ``score(text, context) -> dict`` (or be adaptable to one) and the
    model must be Sonnet-class or stronger.
    """
    if judge_client is None:
        raise NotImplementedError(
            "score_messages requires an LLM judge client (Sonnet+). "
            "Message quality is never assessed by regex (project rule). "
            "Pass a judge client exposing score(text, context) -> "
            "{sentiment, aggression, deception_flag, rationale}."
        )

    events = _read_events(log_path)
    out_path = Path(log_path)
    for e in events:
        if e.get("type") != "observation":
            continue
        obs = e.get("obs", {})
        if obs.get("type") != "message":
            continue
        text = obs.get("text", "")
        verdict = judge_client.score(text, context=obs)
        rec = {
            "type": "msg_score",
            "event_id": e.get("event_id"),
            "scored_by": getattr(judge_client, "name", "judge"),
            "sentiment": float(verdict.get("sentiment", 0.0)),
            "aggression": float(verdict.get("aggression", 0.0)),
            "deception_flag": bool(verdict.get("deception_flag", False)),
            "rationale": str(verdict.get("rationale", "")),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
