#!/usr/bin/env python3
"""Collect a spawned battery's results from the saved call_ids.

The launcher spawns every match to Modal, writes ``jobs.json`` (which records a
``call_id`` per match), and only then starts fetching results in order. If the
local collector dies — killed, disconnected, laptop asleep — the GPU work keeps
running and completes, but nothing writes it down. Re-running the launcher does
NOT recover it: its resume path only reuses logs already on disk, so it would
spawn a second copy of every match and pay twice while the originals finish
into the void.

This script is the recovery path: it reattaches to the existing calls by id and
writes exactly the logs and manifest rows the launcher would have written.
Idempotent — a match whose log is already on disk is skipped, so it can be
re-run against a partially collected directory.

Usage:
    python3 scripts/collect_from_jobs.py data/runs/pg_pilot_v1
    python3 scripts/collect_from_jobs.py data/runs/pg_pilot_v1 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-match .get() timeout in seconds")
    args = ap.parse_args()

    import modal

    run = Path(args.run_dir)
    spec = json.loads((run / "jobs.json").read_text())
    jobs = spec["jobs"]

    mpath = run / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    have = {(r["label"], r["seed"]) for r in manifest}

    def flush():
        mpath.write_text(json.dumps(manifest, indent=2))

    print(f"[collect] {run}: {len(jobs)} spawned, {len(manifest)} already recorded",
          flush=True)

    done = failed = skipped = 0
    for j in jobs:
        key = (j["label"], j["seed"])
        log_path = run / j["label"] / f"seed{j['seed']}.jsonl"
        if key in have or (log_path.exists() and log_path.stat().st_size > 0):
            skipped += 1
            continue
        try:
            res = modal.FunctionCall.from_id(j["call_id"]).get(timeout=args.timeout)
        except Exception as e:
            failed += 1
            print(f"[collect] ERR {j['label']} seed{j['seed']}: "
                  f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(res["log"])
        manifest.append(
            {k: j[k] for k in ("label", "w", "v", "alpha", "seed",
                               "treat_seats", "roles")}
            | {"winner": res["winner"], "rewards": res["rewards"],
               "n_turns": res["n_turns"], "log": str(log_path),
               "treat_seats_echo": res.get("treat_seats"),
               "metrics": res.get("metrics") or {}})
        done += 1
        flush()
        print(f"[collect] {done}/{len(jobs) - skipped} {j['label']} seed{j['seed']} "
              f"winner={res['winner']} turns={res['n_turns']} "
              f"metrics={res.get('metrics')}", flush=True)

    flush()
    print(f"[collect] done: {len(manifest)} rows "
          f"({skipped} already had logs, {done} new, {failed} failed)", flush=True)


if __name__ == "__main__":
    main()
