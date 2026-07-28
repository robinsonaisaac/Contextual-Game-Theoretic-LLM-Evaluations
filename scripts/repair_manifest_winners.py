#!/usr/bin/env python3
"""Re-derive missing manifest fields from the match logs.

The mixed-population launcher's resume path used to write ``winner: null`` for
every REUSED log, so a battery that resumed over earlier matches reports those
matches as unfinished even though they played to a valid ending. The logs
themselves are always authoritative — each carries a ``terminal`` record with
the winner, rewards, and turn count — so the manifest can be repaired offline.

The launcher no longer writes null winners, but any manifest produced by a
resumed run started before that fix needs this pass. It is idempotent and only
ever fills fields in, never overwrites a non-null value.

Usage:
    python3 scripts/repair_manifest_winners.py data/runs/sh_mixed_v2
    python3 scripts/repair_manifest_winners.py data/runs/sh_mixed_v2 --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def terminal_of(log_path: Path):
    """Return the log's terminal record, or None if it never reached one."""
    term = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "terminal":
            term = ev
    return term


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    mpath = run / "manifest.json"
    rows = json.loads(mpath.read_text())

    filled = missing_log = still_unfinished = 0
    for row in rows:
        if row.get("winner"):
            continue
        log_path = Path(row.get("log", ""))
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        if not log_path.exists():
            missing_log += 1
            continue
        term = terminal_of(log_path)
        if term is None or not term.get("winner"):
            still_unfinished += 1
            continue
        row["winner"] = term.get("winner")
        row.setdefault("win_reason", term.get("win_reason", ""))
        if not row.get("rewards"):
            row["rewards"] = term.get("rewards", [])
        if row.get("n_turns") is None:
            row["n_turns"] = term.get("turn")
        if row.get("treat_seats_echo") is None:
            row["treat_seats_echo"] = row.get("treat_seats")
        filled += 1

    done = sum(1 for r in rows if r.get("winner"))
    print(f"{mpath}: {len(rows)} rows | filled {filled} | "
          f"genuinely unfinished {still_unfinished} | missing log {missing_log}")
    print(f"completed after repair: {done}/{len(rows)} ({done / max(len(rows), 1):.1%})")
    if args.dry_run:
        print("--dry-run: nothing written")
        return
    mpath.write_text(json.dumps(rows, indent=2))
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
