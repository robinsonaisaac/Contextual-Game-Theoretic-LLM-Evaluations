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

    `--local-run-dir` should point to the directory that contains
    `index.parquet` and `activations/train/*.pt`. Bundle paths recorded in
    the index are remapped from the volume's `/data/runs/{run_id}/...`
    prefix to the local equivalent.
    """
    from game_theory_llm.steering.storage import (
        load_activation_bundle, read_index, save_vector_set,
    )
    from game_theory_llm.steering.vector_fitting import fit_vectors

    local_run_dir = Path(args.local_run_dir)
    index_path = local_run_dir / "index.parquet"
    df = read_index(index_path, split="train")

    bundles = []
    for row in df.itertuples(index=False):
        rel = Path(row.path).relative_to(f"/data/runs/{args.run_id}")
        local_bundle_path = local_run_dir / rel
        bundles.append(load_activation_bundle(local_bundle_path))

    vs = fit_vectors(bundles, model_name=args.model_name)
    out_path = local_run_dir / "vectors.pt"
    save_vector_set(vs, out_path)
    print(json.dumps({
        "n_vectors": len(vs.vectors),
        "corpus_hash": vs.corpus_hash,
        "vectors_path": str(out_path),
        "n_bundles": len(bundles),
    }, indent=2))


def cmd_push_vectors(args):
    """Push the locally-fit vectors.pt into the Modal Volume."""
    import subprocess
    src = Path(args.local_dir) / "runs" / args.run_id / "vectors.pt"
    dst = f"runs/{args.run_id}/vectors.pt"
    if not src.exists():
        sys.exit(f"vectors.pt not found at {src}")
    subprocess.check_call(["modal", "volume", "put", "--force",
                            "safety", str(src), dst])
    print(f"Uploaded {src} -> volume:{dst}")


def cmd_eval(args):
    import modal
    from game_theory_llm.steering.modal_app import app, SteeringWorker

    prune_stories = _read_jsonl(Path(args.prune_stories))
    sweep_stories = _read_jsonl(Path(args.sweep_stories)) if args.sweep_stories else None
    positions = tuple(args.positions) if args.positions else None
    with app.run():
        worker = SteeringWorker()
        summary = worker.evaluate.remote(
            run_id=args.run_id,
            prune_stories=prune_stories,
            sweep_stories=sweep_stories,
            alpha_prune=args.alpha_prune,
            keep_top_k=args.keep_top_k,
            alpha_grid=tuple(args.alpha_grid),
            layer_stride=args.layer_stride,
            positions=positions,
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
    fit.add_argument("--local-run-dir", required=True,
                     help="Local dir holding index.parquet + activations/train/*.pt")
    fit.add_argument("--model-name", default="gemma-4-e4b-it")
    fit.set_defaults(func=cmd_fit)

    push = sub.add_parser("push-vectors")
    push.add_argument("--run-id", required=True)
    push.add_argument("--local-dir", default="./local_data")
    push.set_defaults(func=cmd_push_vectors)

    ev = sub.add_parser("eval")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--prune-stories", required=True,
                    help="Smaller JSONL used for the dense prune pass.")
    ev.add_argument("--sweep-stories", default=None,
                    help="Larger JSONL used for the alpha sweep on survivors. "
                         "Defaults to --prune-stories if not set.")
    ev.add_argument("--alpha-prune", type=float, default=3.0)
    ev.add_argument("--keep-top-k", type=int, default=5)
    ev.add_argument("--alpha-grid", type=float, nargs="+",
                    default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ev.add_argument("--layer-stride", type=int, default=1)
    ev.add_argument("--positions", nargs="+", default=None,
                    help="Subset of {last_prompt,last_trace,mean_trace} to keep.")
    ev.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
