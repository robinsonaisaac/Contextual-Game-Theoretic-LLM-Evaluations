"""[LEGACY] Single-model variant of the multi-vector + last_trace spawner.

Superseded by ``scripts/run_full_pipeline.py``. Kept for ad-hoc
multi-vector eval combinations not covered by the configured cells.

Spawn the extended sweep: last_trace position + multi-vector combinations.

Single-vector shards use SteeringWorker.eval_shard;
multi-vector shards use SteeringWorker.eval_shard_multi.

Each shard runs detached against the deployed `safety` app.
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
    ap.add_argument("--alpha-grid", default="-2,-1,0,1,2")
    ap.add_argument("--single-cells", default="25:last_trace,27:last_trace",
                    help="Comma-separated layer:position pairs for single-vector shards")
    ap.add_argument("--multi-cells", default="25:mean_trace+27:mean_trace",
                    help="Plus-separated pairs (layer:position+layer:position), "
                         "comma-separated configs")
    ap.add_argument("--call-ids-out", default="local_data/last_eval_extended_call_ids.json")
    ap.add_argument("--single-result-subdir", default="shards")
    ap.add_argument("--multi-result-subdir", default="shards_multi")
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

    SteeringWorker = modal.Cls.from_name(args.app_name, "SteeringWorker")
    worker = SteeringWorker()

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
        "n_stories": len(stories),
        "n_shards": len(shards),
        "shards": shards,
    }, indent=2, default=str))

    print(json.dumps({
        "n_shards_spawned": len(shards),
        "n_stories_each": len(stories),
        "single_cells": single_cells,
        "multi_cells": multi_cells,
        "alphas": alphas,
        "call_ids_out": args.call_ids_out,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
