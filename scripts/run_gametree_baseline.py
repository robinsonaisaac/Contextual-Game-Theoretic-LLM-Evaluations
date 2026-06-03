"""Baseline: does Gemma 4 E4B-it's accuracy fall with game-tree reasoning depth?

Generates depth-stratified minimax game-tree problems (verifiable answers), runs
the model unsteered (eval_shard at alpha=0, reusing the deployed worker), scores
locally by exact numeric match, and prints the depth->accuracy curve. This
establishes whether the synthesized data genuinely tests long-depth reasoning
(accuracy should decay with depth) and validates the generate->verify pipeline
before any fine-tuning.

Usage: python3 scripts/run_gametree_baseline.py [--depths 1,2,3,4,5 --n 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import pandas as pd

from game_theory_llm.reasoning.gametree import gen_game_tree_problem
from game_theory_llm.capability_scoring import gsm8k_correct

HF_ID = "google/gemma-4-E4B-it"
RUN_ID = "reason_gsm8k_v1"     # any run with vectors.pt; alpha=0 => no steering
FRAMINGS = ["abstract", "negotiation", "chess"]


def build(depths, n):
    rows = []
    for d in depths:
        for i in range(n):
            p = gen_game_tree_problem(seed=1000 * d + i, depth=d,
                                      branching=2, framing=FRAMINGS[i % 3])
            rows.append({"story_id": p["story_id"], "prompt": p["prompt"],
                         "coop_choice": "A", "gametree_gold": p["answer"],
                         "depth": d, "seed": 1000 * d + i, "temperature": 0.7,
                         "max_new_tokens": 1024, "game_type": "gametree"})
    out = Path("data/runs/gametree/baseline_corpus.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    rows = build(depths, args.n)
    by_depth = {d: [r for r in rows if r["depth"] == d] for d in depths}
    gold = {r["story_id"]: r["gametree_gold"] for r in rows}
    print(f"[gametree] {len(rows)} problems across depths {depths}")

    worker = modal.Cls.from_name("safety", "SteeringWorker")(model_name=HF_ID)

    def run_depth(d, tries=3):
        stories = by_depth[d]
        for t in range(tries):
            fc = worker.eval_shard.spawn(run_id=RUN_ID, layer=18, position="mean_trace",
                                         alpha=0.0, stories=stories,
                                         result_subdir=f"shards_gametree_d{d}")
            try:
                fc.get(timeout=5400); return d, None
            except Exception as e:
                print(f"  d{d} try{t+1}: {type(e).__name__}", flush=True)
        return d, "failed"

    with ThreadPoolExecutor(max_workers=len(depths)) as ex:
        for fut in as_completed([ex.submit(run_depth, d) for d in depths]):
            d, err = fut.result()
            print(f"[gametree] depth {d} {'ERR' if err else 'done'}", flush=True)

    print("\ndepth  n   accuracy")
    for d in depths:
        local = Path(f"local_data/gametree_d{d}")
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["python3", "-m", "modal", "volume", "get", "--force", "safety",
                        f"runs/{RUN_ID}/shards_gametree_d{d}/", str(local)],
                       capture_output=True)
        pqs = list(local.rglob("*.parquet"))
        if not pqs:
            print(f"{d:>5}  --  MISSING"); continue
        df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        acc = df.apply(lambda r: gsm8k_correct(r["trace"], gold[r["story_id"]]), axis=1)
        print(f"{d:>5} {len(df):>3}  {acc.mean():.3f}")


if __name__ == "__main__":
    main()
