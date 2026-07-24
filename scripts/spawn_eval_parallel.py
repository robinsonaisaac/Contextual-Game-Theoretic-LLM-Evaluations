"""[LEGACY] Single-model parallel shard spawner.

Superseded by ``scripts/run_full_pipeline.py --stage eval``, which uses
configs.py to pick the model and candidate cells. Kept for ad-hoc
single-vector grids.

Fan out eval_shard across N (layer, position, alpha) shards in parallel.

Each shard runs as a separate detached Modal call against the deployed
`safety` app, so the local client can disconnect freely. Aggregating the
shards into a single results_sweep.parquet is done by
scripts/aggregate_shards.py once they're all done.

Usage:
    modal deploy game_theory_llm/steering/modal_app.py   # only if changed
    python3 scripts/spawn_eval_parallel.py \\
        --stories-path data/runs/.../eval.jsonl \\
        --run-id pd_full_v1 \\
        --cells "25:mean_trace,27:mean_trace,30:mean_trace" \\
        --alpha-grid="-2,-1,0,1,2"
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
    ap.add_argument("--cells", required=True,
                    help='Comma-separated layer:position pairs, e.g. "25:mean_trace,27:mean_trace"')
    ap.add_argument("--alpha-grid", default="-2,-1,0,1,2")
    ap.add_argument("--call-ids-out", default="local_data/last_eval_shard_call_ids.json")
    args = ap.parse_args()

    stories = [json.loads(l) for l in Path(args.stories_path).read_text().splitlines() if l.strip()]
    cells = [(int(p.split(":")[0]), p.split(":")[1]) for p in args.cells.split(",")]
    alphas = [float(x) for x in args.alpha_grid.split(",")]

    SteeringWorker = modal.Cls.from_name(args.app_name, "SteeringWorker")
    worker = SteeringWorker()

    shards = []
    for layer, pos in cells:
        for alpha in alphas:
            fc = worker.eval_shard.spawn(
                run_id=args.run_id,
                layer=layer,
                position=pos,
                alpha=alpha,
                stories=stories,
            )
            shards.append({
                "call_id": fc.object_id,
                "layer": layer,
                "position": pos,
                "alpha": alpha,
            })

    Path(args.call_ids_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.call_ids_out).write_text(json.dumps({
        "run_id": args.run_id,
        "n_stories": len(stories),
        "n_shards": len(shards),
        "shards": shards,
    }, indent=2))

    print(json.dumps({
        "n_shards_spawned": len(shards),
        "n_stories_each": len(stories),
        "call_ids_out": args.call_ids_out,
    }, indent=2))


if __name__ == "__main__":
    main()
