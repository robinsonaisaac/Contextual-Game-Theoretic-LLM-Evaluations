"""Local driver for the steering pipeline.

Usage:
    python3 scripts/run_steering.py extract --run-id run1 --stories train.jsonl --split train
    python3 scripts/run_steering.py extract --run-id run1 --stories eval.jsonl  --split eval
    python3 scripts/run_steering.py fit     --run-id run1
    python3 scripts/run_steering.py eval    --run-id run1 --stories eval.jsonl

The `train.jsonl` / `eval.jsonl` files are line-delimited JSON with fields:
    story_id, prompt, coop_choice (one of "A"/"B"), optional seed, temperature.

`extract` and `eval` ship the stories to Modal. `fit` runs locally on the
activation bundles downloaded from the Modal Volume (or a local copy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_extract(args):
    import modal
    from game_theory_llm.steering.modal_app import app, SteeringWorker

    stories = _read_jsonl(Path(args.stories))
    with app.run():
        worker = SteeringWorker()
        summary = worker.extract.remote(stories, args.run_id, args.split)
    print(json.dumps(summary, indent=2))


def cmd_fit(args):
    """Fit vectors locally from per-story bundles in a Modal Volume snapshot.

    Requires that you've previously synced the Modal Volume to local disk:
        modal volume get gtllm-steering runs/{run_id}/ ./local_data/runs/{run_id}/
    """
    from game_theory_llm.steering.storage import (
        load_activation_bundle, read_index, save_vector_set,
    )
    from game_theory_llm.steering.vector_fitting import fit_vectors

    local_run_dir = Path(args.local_dir) / "runs" / args.run_id
    index_path = local_run_dir / "index.parquet"
    df = read_index(index_path, split="train")

    bundles = []
    for row in df.itertuples(index=False):
        bundle = load_activation_bundle(Path(row.path))
        bundles.append(bundle)

    vs = fit_vectors(bundles, model_name=args.model_name)
    out_path = local_run_dir / "vectors.pt"
    save_vector_set(vs, out_path)
    print(json.dumps({
        "n_vectors": len(vs.vectors),
        "corpus_hash": vs.corpus_hash,
        "vectors_path": str(out_path),
    }, indent=2))


def cmd_push_vectors(args):
    """Push the locally-fit vectors.pt into the Modal Volume."""
    import subprocess
    src = Path(args.local_dir) / "runs" / args.run_id / "vectors.pt"
    dst = f"runs/{args.run_id}/vectors.pt"
    if not src.exists():
        sys.exit(f"vectors.pt not found at {src}")
    subprocess.check_call(["modal", "volume", "put", "--force",
                            "gtllm-steering", str(src), dst])
    print(f"Uploaded {src} -> volume:{dst}")


def cmd_eval(args):
    import modal
    from game_theory_llm.steering.modal_app import app, SteeringWorker

    stories = _read_jsonl(Path(args.stories))
    with app.run():
        worker = SteeringWorker()
        summary = worker.evaluate.remote(
            run_id=args.run_id,
            stories=stories,
            alpha_prune=args.alpha_prune,
            keep_top_k=args.keep_top_k,
            alpha_grid=tuple(args.alpha_grid),
        )
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--run-id", required=True)
    ext.add_argument("--stories", required=True)
    ext.add_argument("--split", required=True, choices=["train", "eval"])
    ext.set_defaults(func=cmd_extract)

    fit = sub.add_parser("fit")
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--local-dir", default="./local_data")
    fit.add_argument("--model-name", default="gemma-4-e4b-it")
    fit.set_defaults(func=cmd_fit)

    push = sub.add_parser("push-vectors")
    push.add_argument("--run-id", required=True)
    push.add_argument("--local-dir", default="./local_data")
    push.set_defaults(func=cmd_push_vectors)

    ev = sub.add_parser("eval")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--stories", required=True)
    ev.add_argument("--alpha-prune", type=float, default=3.0)
    ev.add_argument("--keep-top-k", type=int, default=10)
    ev.add_argument("--alpha-grid", type=float, nargs="+",
                    default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ev.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
