"""Fit a 'correctness' steering vector from GSM8k and push it to the volume.

Pipeline (mirrors run_full_pipeline's extract->fit, but correctness-labelled):
  1. Shard the GSM8k-train corpus across K extract workers (generate solution,
     label correct/incorrect via gsm8k_correct, capture residual activations).
  2. rebuild_index over all shard bundles.
  3. Pull bundles locally, mean-difference-fit vectors = mean(correct) -
     mean(incorrect) at every (layer, position), push vectors.pt to the volume.

The resulting run_id (reason_gsm8k_v1) is then swept by run_reasoning_eval.py.

Usage: python3 scripts/run_reasoning_steering.py [--shards 4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

HF_ID = "google/gemma-4-E4B-it"
VOLUME = "safety"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--corpus", default="data/runs/reason_steer/gsm8k_train.jsonl")
    ap.add_argument("--run-id", default="reason_gsm8k_v1")
    args = ap.parse_args()
    global RUN_ID
    RUN_ID = args.run_id

    stories = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    print(f"[reason] {len(stories)} train stories, run_id={RUN_ID}")

    Cls = modal.Cls.from_name("safety", "SteeringWorker")
    worker = Cls(model_name=HF_ID)

    # 1. sharded extract
    chunks = [stories[i::args.shards] for i in range(args.shards)]
    def do_extract(chunk):
        return worker.extract.remote(chunk, RUN_ID, "train")
    print(f"[reason] extracting in {args.shards} shards...")
    with ThreadPoolExecutor(max_workers=args.shards) as ex:
        futs = {ex.submit(do_extract, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            r = fut.result()
            print(f"[reason] shard {futs[fut]} done: n={r['n_stories']} "
                  f"correct_rate={r['coop_rate']:.3f}")

    # 2. rebuild full index over all shard bundles
    rebuild = modal.Function.from_name("safety", "rebuild_index")
    idx = rebuild.remote(RUN_ID, "train")
    print(f"[reason] rebuilt index: {idx['n_bundles']} bundles "
          f"correct={idx['n_coop']} incorrect={idx['n_defect']}")

    # 3. pull bundles + fit locally
    from game_theory_llm.steering.storage import (
        load_activation_bundle, read_index, save_vector_set)
    from game_theory_llm.steering.vector_fitting import fit_vectors

    local_root = Path(f"local_data/{RUN_ID}_dl")
    local_root.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["python3", "-m", "modal", "volume", "get", "--force",
                           VOLUME, f"runs/{RUN_ID}/", str(local_root)])
    run_dir = local_root / RUN_ID if (local_root / RUN_ID).exists() else local_root
    df = read_index(run_dir / "index.parquet", split="train")
    bundles = []
    for row in df.itertuples(index=False):
        rel = Path(row.path).relative_to(f"/data/runs/{RUN_ID}")
        bundles.append(load_activation_bundle(run_dir / rel))
    n_corr = sum(b.cooperated for b in bundles)
    print(f"[reason] fitting on {len(bundles)} bundles "
          f"(correct={n_corr}, incorrect={len(bundles)-n_corr})")
    vs = fit_vectors(bundles, model_name="e4b")
    out = run_dir / "vectors.pt"
    save_vector_set(vs, out)
    subprocess.check_call(["python3", "-m", "modal", "volume", "put", "--force",
                           VOLUME, str(out), f"runs/{RUN_ID}/vectors.pt"])
    print(f"[reason] fitted {len(vs.vectors)} (layer,position) vectors -> "
          f"runs/{RUN_ID}/vectors.pt")
    print("[reason] cells:", sorted(vs.vectors.keys())[:6], "...")


if __name__ == "__main__":
    main()
