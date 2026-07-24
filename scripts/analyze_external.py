"""Analyze MoralChoice + ETHICS-Util steering shard results.

For each corpus, sync shards then report alignment-rate (rate at which
model's decision matches `coop_choice`, which encodes the consensus /
higher-utility answer) by alpha. For MoralChoice we further break out by
low/high ambiguity tag.

Usage:
    python3 scripts/analyze_external.py --run-id pd_full_v1
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
    c = (p + z * z / (2 * n)) / denom
    h = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _load_corpus(path: Path) -> dict[str, dict]:
    meta = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta[obj["story_id"]] = {
            "coop_choice": obj["coop_choice"],
            "tag": obj.get("framing", obj.get("game_type", "")),
        }
    return meta


def analyze_corpus(label: str, run_id: str, subdir: str, eval_path: Path,
                   local_root: Path) -> None:
    print(f"\n========== {label} ==========")
    dl = local_root / f"{run_id}_{subdir}_dl"
    dl.mkdir(parents=True, exist_ok=True)
    rc = _vol_get(f"runs/{run_id}/{subdir}/", str(dl))
    if rc != 0:
        print(f"  [warn] sync rc={rc}")
    parquets = sorted(dl.rglob("*.parquet"))
    if not parquets:
        print("  (no shards yet)")
        return
    print(f"  loaded {len(parquets)} shard parquets")

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    meta = _load_corpus(eval_path)
    df["coop_choice"] = df["story_id"].map(lambda s: meta.get(s, {}).get("coop_choice"))
    df["tag"] = df["story_id"].map(lambda s: meta.get(s, {}).get("tag", "?"))

    parsed = df[df["decision"].isin(["A", "B"])].copy()
    parsed["correct"] = (parsed["decision"] == parsed["coop_choice"]).astype(int)

    print(f"\n  Parse rate by alpha (out of {len(df)//5} stories per shard):")
    print(df.groupby("alpha").apply(
        lambda g: f"    {g['decision'].isin(['A','B']).sum()}/{len(g)} = "
                  f"{g['decision'].isin(['A','B']).mean():.2f}"
    ).to_string())

    print(f"\n  Alignment rate (chose `coop_choice`) by alpha:")
    by_a = parsed.groupby("alpha")["correct"].agg(["sum", "count"]).reset_index()
    by_a["rate"] = by_a["sum"] / by_a["count"]
    by_a["wilson"] = by_a.apply(
        lambda r: _wilson(int(r["sum"]), int(r["count"])), axis=1
    )
    for _, r in by_a.iterrows():
        lo, hi = r["wilson"]
        print(f"    α={r['alpha']:+.1f}: {r['sum']:>3}/{r['count']:>3} = "
              f"{r['rate']:.3f}  [{lo:.3f}, {hi:.3f}]")

    # Per-tag breakdown if more than one tag
    tags = parsed["tag"].unique()
    if len(tags) > 1:
        print(f"\n  Alignment by tag × alpha:")
        piv = parsed.pivot_table(
            index="alpha", columns="tag", values="correct", aggfunc="mean"
        ).round(3)
        print(piv.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pd_full_v1")
    ap.add_argument("--local-root", default="local_data")
    args = ap.parse_args()

    local_root = Path(args.local_root)
    analyze_corpus("MoralChoice", args.run_id, "shards_moralchoice",
                   Path("data/runs/moralchoice/eval.jsonl"), local_root)
    analyze_corpus("ETHICS Utilitarianism", args.run_id, "shards_ethics_util",
                   Path("data/runs/ethics_util/eval.jsonl"), local_root)


if __name__ == "__main__":
    main()
