#!/usr/bin/env python3
"""Evaluate the existing sharp-run stories with Gemma-family models.

AAAI-27 revision (2026-07-24): NeurIPS reviewers' most-repeated criticism was
that the mechanistic/steering sections use Gemma while the Section-3
behavioral battery does not, so the paper's two halves never connect. This
runs the SAME 3,010 stories through OpenRouter-hosted Gemma 4 variants with
the SAME harness as the frontier eval (`run_evaluation` with a custom model
dict), writing to ``evals_gemma/`` beside ``evals/``. Idempotent: re-running
skips already-scored (cell, story, model) tuples.

Gemma 4 E4B-it (the paper's primary steering model) is not OpenRouter-hosted;
it is evaluated separately on the Modal "safety" workers at alpha=0.

Usage:
    python3 scripts/run_gemma_evals.py [--run-id 2026-05-05-sharp] [--dry-run]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game_theory_llm.client import LLMClient, MODEL_REGISTRY
from game_theory_llm.runner import ProductionRun, build_cells, cell_id

GEMMA_MODELS = {
    "gemma-4-26b-a4b-it": MODEL_REGISTRY["gemma"]["gemma-4-26b-a4b-it"],
    "gemma-4-31b-it":     MODEL_REGISTRY["gemma"]["gemma-4-31b-it"],
}


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="2026-05-05-sharp")
    p.add_argument("--root", default="data/runs")
    p.add_argument("--eval-concurrency", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run = ProductionRun(run_id=args.run_id, root=args.root,
                        eval_concurrency=args.eval_concurrency)
    cells = build_cells()
    stories_by_cell = {}
    for c in cells:
        cid = cell_id(c.game_config.id, c.contrast_dim, getattr(c, c.contrast_dim))
        stories_by_cell[cid] = run._existing_stories(c)
    n_stories = sum(len(v) for v in stories_by_cell.values())
    out_dir = run.run_dir / "evals_gemma"

    print(f"Run ID:          {args.run_id}")
    print(f"Cells:           {len(cells)}")
    print(f"Stories on disk: {n_stories}")
    print(f"Gemma models ({len(GEMMA_MODELS)}): {', '.join(GEMMA_MODELS)}")
    print(f"Total evals:     {n_stories * len(GEMMA_MODELS):,}")
    print(f"Output dir:      {out_dir}")

    if args.dry_run:
        print("\nDry-run — exiting without API calls.")
        return

    client = LLMClient(models=dict(GEMMA_MODELS))
    asyncio.run(run.run_evaluation(client, stories_by_cell, cells,
                                   models=GEMMA_MODELS, out_dir=out_dir))
    n_files = len(list(out_dir.glob("*.jsonl")))
    print(f"Done: {n_files} eval files in {out_dir}")


if __name__ == "__main__":
    main()
