"""[LEGACY] Single-model spawn for the prune-then-sweep evaluate() entrypoint.

For new work prefer ``scripts/run_full_pipeline.py`` which orchestrates
download → extract → fit → eval → aggregate across all configured models.
This script is kept for ad-hoc one-off prune/sweep experiments on a
single (run_id, vector_set).

Spawn a detached evaluate() call against the deployed `safety` Modal app.

The function call runs on Modal's cloud and persists even if the local
client disconnects (which killed our 3.6 hr eval the first time). The
returned function-call ID can be used later to fetch the result, but the
worker also writes incremental progress to /data/runs/{run_id}/eval_progress.jsonl
on the volume.

Workflow:
    1. modal deploy game_theory_llm/steering/modal_app.py
    2. python3 scripts/spawn_eval.py \\
           --prune-path data/runs/.../prune.jsonl \\
           --sweep-path data/runs/.../prune.jsonl \\
           --run-id pd_full_v1 \\
           --survivor-override "25:mean_trace,27:mean_trace,30:mean_trace" \\
           --alpha-grid "-2,-1,0,1,2"
    3. python3 scripts/check_eval.py --call-id <printed-id> --run-id pd_full_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


def parse_override(s: str) -> tuple[tuple[int, str], ...] | None:
    if not s:
        return None
    return tuple(
        (int(x.split(":")[0]), x.split(":")[1])
        for x in s.split(",")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-name", default="safety")
    ap.add_argument("--prune-path", required=True)
    ap.add_argument("--sweep-path", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--alpha-prune", type=float, default=3.0)
    ap.add_argument("--keep-top-k", type=int, default=5)
    ap.add_argument("--alpha-grid", default="-2,-1,0,1,2")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--positions", default="mean_trace")
    ap.add_argument("--survivor-override", default="",
                    help="Comma-separated layer:position pairs, e.g. '25:mean_trace,27:mean_trace'")
    ap.add_argument("--call-id-out", default="local_data/last_eval_call_id.txt")
    args = ap.parse_args()

    prune_stories = [json.loads(l) for l in Path(args.prune_path).read_text().splitlines() if l.strip()]
    sweep_stories = [json.loads(l) for l in Path(args.sweep_path).read_text().splitlines() if l.strip()]
    grid = tuple(float(x) for x in args.alpha_grid.split(","))
    pos_tuple = tuple(p.strip() for p in args.positions.split(",")) if args.positions else None
    override = parse_override(args.survivor_override)

    SteeringWorker = modal.Cls.from_name(args.app_name, "SteeringWorker")
    worker = SteeringWorker()
    fc = worker.evaluate.spawn(
        run_id=args.run_id,
        prune_stories=prune_stories,
        sweep_stories=sweep_stories,
        alpha_prune=args.alpha_prune,
        keep_top_k=args.keep_top_k,
        alpha_grid=grid,
        layer_stride=args.layer_stride,
        positions=pos_tuple,
        survivor_override=override,
    )

    Path(args.call_id_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.call_id_out).write_text(fc.object_id + "\n")

    print(json.dumps({
        "call_id": fc.object_id,
        "run_id": args.run_id,
        "n_prune_stories": len(prune_stories),
        "n_sweep_stories": len(sweep_stories),
        "survivor_override": override,
        "call_id_saved_to": args.call_id_out,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
