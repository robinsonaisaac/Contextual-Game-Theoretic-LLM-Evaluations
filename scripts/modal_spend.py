#!/usr/bin/env python3
"""Estimate Modal GPU spend for steering-game runs from their JSONL logs.

Every log record carries a second-resolution ``ts``; a match's wall time is
last_ts - first_ts, and matches run one-per-container-call on the "safety"
app's SteeringWorker (gpu=A100-80GB, scaledown_window=300s). Billable time
exceeds in-match wall time by model-load/cold-start and idle scaledown
windows, which the logs cannot see — ``--overhead-frac`` (default 0.25)
covers that; treat output as an ESTIMATE and reconcile against the
authoritative number at https://modal.com → workspace → Usage.

Usage:
    python3 scripts/modal_spend.py data/runs/sh_steering_v2 [more run dirs]
    python3 scripts/modal_spend.py --all          # every data/runs/* dir
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# $/hr; update from https://modal.com/pricing if rates change.
GPU_RATES = {"A100-80GB": 2.50}
WORKER_GPU = "A100-80GB"


def _match_seconds(log: Path) -> float:
    first = last = None
    with log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ts = json.loads(line).get("ts")
            except json.JSONDecodeError:
                continue
            if not ts:
                continue
            if first is None:
                first = ts
            last = ts
    if not first or not last:
        return 0.0
    fmt = "%Y-%m-%dT%H:%M:%S"
    return max(
        0.0,
        (datetime.strptime(last, fmt) - datetime.strptime(first, fmt)).total_seconds(),
    )


def run_dir_estimate(run_dir: Path, overhead_frac: float) -> dict:
    logs = sorted(run_dir.glob("*/seed*.jsonl")) or sorted(run_dir.glob("seed*.jsonl"))
    secs = sum(_match_seconds(p) for p in logs)
    gpu_hours = secs / 3600.0 * (1.0 + overhead_frac)
    rate = GPU_RATES[WORKER_GPU]
    return {
        "run": run_dir.name,
        "matches": len(logs),
        "match_hours": round(secs / 3600.0, 2),
        "est_gpu_hours": round(gpu_hours, 2),
        "est_usd": round(gpu_hours * rate, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true",
                    help="scan every directory under data/runs/")
    ap.add_argument("--overhead-frac", type=float, default=0.25,
                    help="fractional uplift for cold-start + idle windows")
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.all:
        dirs = sorted(p for p in Path("data/runs").iterdir() if p.is_dir())
    if not dirs:
        ap.error("pass run dirs or --all")

    rows = [run_dir_estimate(d, args.overhead_frac) for d in dirs]
    rows = [r for r in rows if r["matches"]]
    hdr = f"{'run':28s} {'matches':>7} {'matchHrs':>9} {'estGPUh':>8} {'est$':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:28s} {r['matches']:>7} {r['match_hours']:>9.2f} "
              f"{r['est_gpu_hours']:>8.2f} {r['est_usd']:>8.2f}")
    tot_h = sum(r["est_gpu_hours"] for r in rows)
    tot_d = sum(r["est_usd"] for r in rows)
    print("-" * len(hdr))
    print(f"{'TOTAL':28s} {sum(r['matches'] for r in rows):>7} "
          f"{sum(r['match_hours'] for r in rows):>9.2f} {tot_h:>8.2f} {tot_d:>8.2f}")
    print(f"\nrate: {WORKER_GPU} @ ${GPU_RATES[WORKER_GPU]:.2f}/hr, "
          f"overhead +{args.overhead_frac:.0%}. Estimate only — reconcile at "
          "modal.com workspace Usage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
