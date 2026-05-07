"""Pull all per-shard parquets from the volume and stitch into one results file."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--volume-name", default="safety")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    local_shard_dir = Path(f"local_data/{args.run_id}_shards")
    local_shard_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python3", "-m", "modal", "volume", "get", "--force",
        args.volume_name, f"runs/{args.run_id}/shards/", str(local_shard_dir),
    ])

    pqs = sorted(local_shard_dir.rglob("*.parquet"))
    print(f"merging {len(pqs)} parquet shards")
    dfs = [pd.read_parquet(p) for p in pqs]
    big = pd.concat(dfs, ignore_index=True)

    out = Path(args.out or f"local_data/{args.run_id}/results_sweep.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out, index=False)

    agg = big.groupby(["layer", "position", "alpha"]).agg(
        n=("story_id", "count"),
        coop_rate=("cooperated", "mean"),
        n_unparsed=("decision", lambda s: ((s == "") | s.isna()).sum()),
    ).reset_index()
    print(agg.to_string(index=False))
    print(f"\nwrote {out} ({len(big)} rows)")


if __name__ == "__main__":
    main()
