"""[LEGACY] Manual multi-model spawn — superseded by run_full_pipeline.py.

For new work prefer ``scripts/run_full_pipeline.py --model {name} --stage eval``,
which sources cells/alphas from ``configs.py`` instead of from CLI flags.
This script is kept for ad-hoc cell sets not in the registry.

Picks SteeringWorker (A100-40GB) or SteeringWorkerLarge (A100-80GB) based
on a --gpu-tier flag, and routes shards through the right class with the
right model_name.

Usage:
    python3 scripts/spawn_eval_extended_multi.py \\
        --stories-path data/runs/.../full_corpus.jsonl \\
        --run-id pd_E2B_v1 \\
        --model-name google/gemma-4-E2B-it \\
        --gpu-tier small \\
        --single-cells "12:mean_trace,16:mean_trace" \\
        --alpha-grid="-3,-2,-1,0,1,2,3"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-name", default="safety")
    ap.add_argument("--stories-path", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-name", required=True,
                    help="HF id, e.g. google/gemma-4-E2B-it")
    ap.add_argument("--gpu-tier", choices=["small", "large"], required=True,
                    help="'small' = A100-40GB (≤7B), 'large' = A100-80GB (>10B)")
    ap.add_argument("--alpha-grid", default="-3,-2,-1,0,1,2,3")
    ap.add_argument("--single-cells", default="",
                    help="Comma-separated layer:position pairs")
    ap.add_argument("--multi-cells", default="",
                    help="Plus-separated pairs, comma-separated configs")
    ap.add_argument("--single-result-subdir", default="shards")
    ap.add_argument("--multi-result-subdir", default="shards_multi")
    ap.add_argument("--call-ids-out", required=True)
    args = ap.parse_args()

    stories = [json.loads(l) for l in Path(args.stories_path).read_text().splitlines() if l.strip()]
    alphas = [float(x) for x in args.alpha_grid.split(",")]
    single_cells = [
        (int(p.split(":")[0]), p.split(":")[1])
        for p in args.single_cells.split(",")
        if p.strip()
    ]
    multi_cells = []
    if args.multi_cells:
        for combo in args.multi_cells.split(","):
            cells = [
                (int(p.split(":")[0]), p.split(":")[1])
                for p in combo.split("+")
            ]
            multi_cells.append(cells)

    cls_name = "SteeringWorker" if args.gpu_tier == "small" else "SteeringWorkerLarge"
    Cls = modal.Cls.from_name(args.app_name, cls_name)
    worker = Cls(model_name=args.model_name)

    shards = []
    for layer, pos in single_cells:
        for alpha in alphas:
            fc = worker.eval_shard.spawn(
                run_id=args.run_id,
                layer=layer,
                position=pos,
                alpha=alpha,
                stories=stories,
                result_subdir=args.single_result_subdir,
            )
            shards.append({
                "kind": "single",
                "call_id": fc.object_id,
                "layer": layer,
                "position": pos,
                "alpha": alpha,
            })
    for cells in multi_cells:
        label = "+".join(f"L{l}_{p}" for l, p in cells)
        for alpha in alphas:
            fc = worker.eval_shard_multi.spawn(
                run_id=args.run_id,
                cells=cells,
                alpha=alpha,
                stories=stories,
                label=label,
                result_subdir=args.multi_result_subdir,
            )
            shards.append({
                "kind": "multi",
                "call_id": fc.object_id,
                "cells": cells,
                "label": label,
                "alpha": alpha,
            })

    Path(args.call_ids_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.call_ids_out).write_text(json.dumps({
        "run_id": args.run_id,
        "model_name": args.model_name,
        "gpu_tier": args.gpu_tier,
        "n_stories": len(stories),
        "n_shards": len(shards),
        "shards": shards,
    }, indent=2, default=str))

    print(json.dumps({
        "model_name": args.model_name,
        "gpu_tier": args.gpu_tier,
        "n_shards_spawned": len(shards),
        "n_stories_each": len(stories),
        "call_ids_out": args.call_ids_out,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
