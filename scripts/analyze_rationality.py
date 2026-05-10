"""Analyze rationality-steering eval results.

Pulls per-shard parquets from the safety volume, joins with story metadata
(specifically `game_type`), and reports per-game B-rate vs steering
coefficient $\\alpha$.

For each held-out game we expect:
    stag_hunt           : positive alpha -> higher B-rate (risk-dominant Hare)
    chicken             : ambiguous (no clean dominant strategy)
    battle_of_the_sexes : ambiguous

Usage:
    python3 scripts/analyze_rationality.py --run-id rat_E4B_v1 \\
        --eval-stories data/runs/2026-05-05-sharp/rationality/heldout_eval_subset.jsonl \\
        --result-subdir shards
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _vol_get(remote: str, local: str) -> int:
    return subprocess.call(
        ["python3", "-m", "modal", "volume", "get", "--force",
         "safety", remote, local],
        stderr=subprocess.DEVNULL,
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--eval-stories", required=True,
                    help="Same JSONL used for the eval (provides game_type per story_id).")
    ap.add_argument("--result-subdir", default="shards")
    ap.add_argument("--local-root", default="local_data")
    args = ap.parse_args()

    local_root = Path(args.local_root) / f"{args.run_id}_results"
    local_root.mkdir(parents=True, exist_ok=True)
    rc = _vol_get(f"runs/{args.run_id}/{args.result_subdir}/", str(local_root))
    if rc != 0:
        sys.exit(f"sync failed (rc={rc})")

    parquets = list(local_root.rglob("*.parquet"))
    if not parquets:
        sys.exit(f"no parquets under {local_root}")
    print(f"[load] {len(parquets)} shard parquets", flush=True)

    rows = []
    for p in parquets:
        df = pd.read_parquet(p)
        rows.append(df)
    raw = pd.concat(rows, ignore_index=True)

    # Story metadata: game_type per story_id
    meta = {}
    for line in Path(args.eval_stories).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta[obj["story_id"]] = {"game_type": obj["game_type"], "framing": obj.get("framing", "")}
    if "story_id" not in raw.columns:
        sys.exit("expected story_id column in shards")
    raw["game_type"] = raw["story_id"].map(lambda sid: meta.get(sid, {}).get("game_type", "?"))

    # Per (game, alpha): B-rate (decision == 'B') with Wilson CI.
    # The pipeline stored per-story decisions in 'decision' (A/B) and
    # 'cooperated' (decision == coop_choice). For held-out we set
    # coop_choice='B' so cooperation_rate IS B-rate, but to be robust we
    # recompute from `decision` directly.
    raw["b_choice"] = (raw["decision"].astype(str).str.upper() == "B").astype(int)
    raw["a_choice"] = (raw["decision"].astype(str).str.upper() == "A").astype(int)

    grouped = raw.groupby(["game_type", "alpha"]).agg(
        n=("decision", "count"),
        n_b=("b_choice", "sum"),
        n_a=("a_choice", "sum"),
    ).reset_index()
    grouped["b_rate"] = grouped["n_b"] / grouped["n"]
    grouped["a_rate"] = grouped["n_a"] / grouped["n"]

    cis = grouped.apply(lambda r: _wilson_ci(int(r["n_b"]), int(r["n"])), axis=1)
    grouped["b_lo"] = [c[0] for c in cis]
    grouped["b_hi"] = [c[1] for c in cis]

    print("\n=== Rationality steering: B-rate by game and alpha ===")
    pivot = grouped.pivot_table(
        index="alpha", columns="game_type",
        values="b_rate"
    ).round(3)
    print(pivot.to_string())

    print("\n=== Per-game alpha=-3 vs alpha=+3 (Δ B-rate) ===")
    span = grouped.pivot_table(
        index="game_type", columns="alpha",
        values="b_rate"
    )
    if -3.0 in span.columns and 3.0 in span.columns:
        span["delta"] = span[3.0] - span[-3.0]
        print(span[[-3.0, 0.0, 3.0, "delta"]].round(3).to_string())

    out = local_root.parent / f"{args.run_id}_summary.parquet"
    grouped.to_parquet(out)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
