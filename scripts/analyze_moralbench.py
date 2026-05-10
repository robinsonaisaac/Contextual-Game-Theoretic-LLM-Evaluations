"""Analyze MoralBench steering results.

Pulls per-shard parquets from the safety volume and reports per-foundation
Agree-rate (A-rate) vs steering coefficient $\\alpha$.

If the cooperation direction is real, we expect Care/Harm and Fairness to
shift more strongly with $\\alpha$ than the binding foundations
(Loyalty/Authority/Sanctity).

Usage:
    python3 scripts/analyze_moralbench.py --run-id pd_full_v1 \\
        --subdir shards_moralbench \\
        --eval-stories data/runs/moralbench/eval.jsonl
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


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pd_full_v1")
    ap.add_argument("--subdir", default="shards_moralbench")
    ap.add_argument("--eval-stories", required=True)
    ap.add_argument("--local-root", default="local_data")
    args = ap.parse_args()

    local_root = Path(args.local_root) / f"{args.run_id}_{args.subdir}_dl"
    local_root.mkdir(parents=True, exist_ok=True)
    rc = _vol_get(f"runs/{args.run_id}/{args.subdir}/", str(local_root))
    if rc != 0:
        sys.exit(f"sync failed (rc={rc})")

    parquets = sorted(local_root.rglob("*.parquet"))
    if not parquets:
        sys.exit(f"no parquets under {local_root}")
    print(f"[load] {len(parquets)} shard parquets", flush=True)

    raw = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    # Story metadata: foundation per story_id
    meta = {}
    for line in Path(args.eval_stories).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta[obj["story_id"]] = {
            "foundation": obj["framing"],
            "dataset": obj["game_type"],
            "score_A": obj.get("human_score_A"),
            "score_B": obj.get("human_score_B"),
        }
    raw["foundation"] = raw["story_id"].map(lambda s: meta.get(s, {}).get("foundation", "?"))
    raw["dataset"] = raw["story_id"].map(lambda s: meta.get(s, {}).get("dataset", "?"))

    parsed = raw[raw["decision"].isin(["A", "B"])].copy()
    parsed["agree"] = (parsed["decision"] == "A").astype(int)

    print()
    print(f"=== Parse rate by alpha ({len(raw)} total stories per shard) ===")
    pr = raw.groupby("alpha").apply(
        lambda g: f"{(g['decision'].isin(['A','B'])).sum()}/{len(g)} = "
                  f"{(g['decision'].isin(['A','B'])).mean():.2f}"
    )
    print(pr.to_string())

    print()
    print("=== Agree-rate (A-rate) by foundation × alpha (parsed only) ===")
    piv = parsed.pivot_table(
        index="alpha", columns="foundation", values="agree", aggfunc="mean"
    ).round(3)
    print(piv.to_string())

    print()
    print("=== Sample sizes (parsed N per cell) ===")
    ns = parsed.pivot_table(
        index="alpha", columns="foundation", values="agree", aggfunc="count"
    )
    print(ns.to_string())

    print()
    print("=== Δ Agree-rate (alpha=+3 minus alpha=-3) by foundation ===")
    a3 = parsed[parsed["alpha"] == 3.0].groupby("foundation")["agree"].mean()
    am3 = parsed[parsed["alpha"] == -3.0].groupby("foundation")["agree"].mean()
    delta = (a3 - am3).round(3).rename("delta")
    n_a3 = parsed[parsed["alpha"] == 3.0].groupby("foundation")["agree"].count().rename("n_+3")
    n_am3 = parsed[parsed["alpha"] == -3.0].groupby("foundation")["agree"].count().rename("n_-3")
    table = pd.concat(
        [am3.rename("a-3"), a3.rename("a+3"), delta, n_am3, n_a3], axis=1
    ).round(3)
    print(table.to_string())

    out = local_root.parent / f"{args.run_id}_moralbench_summary.parquet"
    parsed.to_parquet(out)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
