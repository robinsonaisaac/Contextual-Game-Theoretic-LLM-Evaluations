"""Sweep the GSM8k-fit 'correctness' steering vector and test for a reasoning
LIFT on held-out GSM8k (and optionally GPQA).

For each (layer, alpha) it runs an eval shard with the correctness vector
(run_id reason_gsm8k_v1) on the held-out GSM8k eval corpus, then pulls the
generation traces from the volume and scores them locally (exact numeric
match). alpha=0 is the shared unsteered baseline; +alpha pushes toward
'correct-like' activations (the hypothesis), -alpha is the control (should not
help, ideally hurts, validating the axis).

Usage:
    python3 scripts/run_reasoning_eval.py                       # default grid
    python3 scripts/run_reasoning_eval.py --layers 18,22 --alphas " -6,0,6"
    python3 scripts/run_reasoning_eval.py --eval gpqa --no-spawn # score only
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import pandas as pd
from scipy.stats import fisher_exact

from game_theory_llm.capability_scoring import gsm8k_correct

HF_ID = "google/gemma-4-E4B-it"
POSITION = "mean_trace"
VOLUME = "safety"
EVAL_PATHS = {"gsm8k": "data/runs/capability/gsm8k_eval.jsonl",
              "gpqa": "data/runs/reasoning/gpqa_diamond_eval.jsonl",
              "mmlu": "data/runs/reasoning/mmlu_eval.jsonl",
              "bbh": "data/runs/bbh/logical_deduction_eval.jsonl"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    import math
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=list(EVAL_PATHS), default="gsm8k")
    ap.add_argument("--layers", default="14,18,22,26")
    ap.add_argument("--alphas", default="-6,0,6")
    ap.add_argument("--run-id", default="reason_gsm8k_v1")
    ap.add_argument("--no-spawn", action="store_true", help="skip eval, just pull+score")
    args = ap.parse_args()
    global RUN_ID
    RUN_ID = args.run_id

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    corpus = {json.loads(l)["story_id"]: json.loads(l)
              for l in Path(EVAL_PATHS[args.eval]).read_text().splitlines() if l.strip()}
    stories = list(corpus.values())
    subdir = f"shards_reval_{args.eval}"

    # (layer, alpha) grid; alpha=0 baseline once (steering is a no-op).
    cells = []
    if 0.0 in alphas:
        cells.append((layers[0], 0.0))
    for L in layers:
        for a in alphas:
            if a != 0.0:
                cells.append((L, a))

    if not args.no_spawn:
        worker = modal.Cls.from_name("safety", "SteeringWorker")(model_name=HF_ID)
        print(f"[reval] spawning {len(cells)} cells on {args.eval} (n={len(stories)})")
        manifest = []
        for (L, a) in cells:
            fc = worker.eval_shard.spawn(run_id=RUN_ID, layer=L, position=POSITION,
                                         alpha=a, stories=stories, result_subdir=subdir)
            manifest.append({"layer": L, "alpha": a, "call_id": fc.object_id})

        def fetch(m):
            try:
                modal.FunctionCall.from_id(m["call_id"]).get(timeout=5400); return m, None
            except Exception as e:
                return m, f"{type(e).__name__}: {e}"
        with ThreadPoolExecutor(max_workers=16) as ex:
            for fut in as_completed([ex.submit(fetch, m) for m in manifest]):
                m, err = fut.result()
                print(f"[reval] {'ERR '+err if err else 'done'}: L{m['layer']} a={m['alpha']:+g}")

    # pull shards + score locally
    local = Path(f"local_data/{RUN_ID}_{subdir}")
    local.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["python3", "-m", "modal", "volume", "get", "--force",
                           VOLUME, f"runs/{RUN_ID}/{subdir}/", str(local)])
    df = pd.concat([pd.read_parquet(p) for p in local.rglob("*.parquet")], ignore_index=True)

    def correct(r):
        if args.eval == "gsm8k":
            return gsm8k_correct(r["trace"], corpus[r["story_id"]]["gsm8k_gold"])
        return r["story_id"] in corpus and r["decision"] == corpus[r["story_id"]]["coop_choice"]
    df["correct"] = df.apply(correct, axis=1)

    base = df[df["alpha"] == 0.0]
    bk, bn = int(base["correct"].sum()), len(base)
    print(f"\n=== reasoning steering on {args.eval} (baseline n={bn}, acc={bk/bn:.3f}) ===")
    print(f"{'layer':>5} {'alpha':>6} {'n':>4} {'acc':>7} {'95% CI':>15} {'d_vs_base':>10} {'p':>7}")
    rows = []
    for (L, a), g in sorted(df.groupby(["layer", "alpha"])):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, n)
        d = k/n - bk/bn
        p = fisher_exact([[k, n-k], [bk, bn-bk]])[1] if a != 0.0 else None
        rows.append({"layer": int(L), "alpha": float(a), "n": n, "acc": k/n,
                     "ci_lo": lo, "ci_hi": hi, "delta_vs_base": d, "fisher_p": p})
        ps = f"{p:.3f}" if p is not None else " base"
        print(f"{int(L):>5} {a:>+6g} {n:>4} {k/n:>7.3f} [{lo:.2f},{hi:.2f}] {d:>+10.3f} {ps:>7}")
    out = Path(f"data/runs/reason_steer/results_{args.eval}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline_acc": bk/bn, "baseline_n": bn, "cells": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
