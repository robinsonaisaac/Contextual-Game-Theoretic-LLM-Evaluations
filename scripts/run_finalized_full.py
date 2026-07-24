#!/usr/bin/env python3
"""Run the FINAL multi-model steering pipeline at maximum parallelism.

Phases (each phase is parallel within itself):

  1. Extract train traces + activations on every model in parallel.
     (3 detached Modal spawns; we then block on all of them.)
  2. Sync bundles to local, fit + push vectors per model (CPU only, fast).
  3. Spawn all eval shards for both regular + label-swap corpora across
     every model in parallel. ((2 layers × 7 alphas × 2 corpora × 3 models)
     = 84 single-vector shards; plus any multi_cells from configs.)
  4. Block until every eval shard finishes.
  5. Aggregate per-model parquets locally.

Run-IDs use the suffix ``--run-tag`` (default: ``final``); old ``_v1`` data
on the volume is left in place as the implicit archive.

Usage:
    python3 -m modal deploy game_theory_llm/steering/modal_app.py
    python3 scripts/run_finalized_full.py
    # or with a custom tag / smaller story set:
    python3 scripts/run_finalized_full.py --run-tag v3 \
        --train-stories data/.../train_subset.jsonl \
        --eval-stories  data/.../full_corpus.jsonl \
        --swap-stories  data/.../full_corpus_swap.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `scripts/` importable so we can reuse run_full_pipeline's stage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_theory_llm.steering.configs import MODEL_CONFIGS, run_id_for
from run_full_pipeline import (  # type: ignore
    stage_download, stage_warmup, stage_extract, stage_extract_wait,
    stage_fit, stage_eval, stage_eval_wait, stage_eval_status, stage_aggregate,
    _modal_volume_get, DEFAULT_ALPHA_GRID,
)


def _ensure_local_dir(name: str) -> Path:
    p = Path("local_data") / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="final")
    ap.add_argument("--train-stories",
                    default="data/runs/2026-05-05-sharp/steering/train.jsonl")
    ap.add_argument("--eval-stories",
                    default="data/runs/2026-05-05-sharp/steering/full_corpus.jsonl")
    ap.add_argument("--swap-stories",
                    default="data/runs/2026-05-05-sharp/steering/full_corpus_swap.jsonl")
    ap.add_argument("--alpha-grid",
                    default=",".join(str(a) for a in DEFAULT_ALPHA_GRID))
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip stage 0 (assumes weights already on volume).")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip stage 1 + 2 (assumes vectors.pt already on volume).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Spawn nothing; only aggregate already-completed shards.")
    args = ap.parse_args()

    alpha_grid = tuple(float(x) for x in args.alpha_grid.split(","))
    train_stories = Path(args.train_stories)
    eval_stories = Path(args.eval_stories)
    swap_stories = Path(args.swap_stories)
    if not train_stories.exists():
        sys.exit(f"missing {train_stories}")
    if not eval_stories.exists():
        sys.exit(f"missing {eval_stories}")
    if not swap_stories.exists():
        sys.exit(f"missing {swap_stories}")

    models = list(MODEL_CONFIGS.values())
    run_ids = {m.short_name: run_id_for(m.short_name, args.run_tag) for m in models}
    print(f"[finalized] tag={args.run_tag}; run_ids={run_ids}", flush=True)

    # ------ Phase 0: download (cheap, idempotent) ------
    if not args.skip_download:
        for m in models:
            stage_download(m)

    # ------ Phase 1: spawn all extracts in parallel ------
    if not args.skip_extract:
        extract_calls: dict[str, str] = {}
        for m in models:
            extract_calls[m.short_name] = stage_extract(m, run_ids[m.short_name], train_stories)
        Path("local_data/finalized_extract_calls.json").write_text(
            json.dumps({"tag": args.run_tag, "calls": extract_calls}, indent=2)
        )
        # ------ Phase 1b: block on all extracts ------
        for name, cid in extract_calls.items():
            print(f"[finalized] waiting on extract {name}...", flush=True)
            stage_extract_wait(cid)

        # ------ Phase 2: fit + push per model (sequential local CPU work) ------
        for m in models:
            stage_fit(m, run_ids[m.short_name])

    # ------ Phase 3: spawn all evals (regular + swap) in parallel ------
    if not args.skip_eval:
        for m in models:
            print(f"[finalized] spawning regular eval for {m.short_name}...", flush=True)
            stage_eval(
                m, run_ids[m.short_name], eval_stories,
                alpha_grid=alpha_grid,
                result_subdir="shards",
                multi_result_subdir="shards_multi",
                call_ids_filename="eval_call_ids_regular.json",
            )
        for m in models:
            print(f"[finalized] spawning SWAP eval for {m.short_name}...", flush=True)
            stage_eval(
                m, run_ids[m.short_name], swap_stories,
                alpha_grid=alpha_grid,
                result_subdir="shards_swap",
                multi_result_subdir="shards_multi_swap",
                call_ids_filename="eval_call_ids_swap.json",
            )

        # ------ Phase 4: block on every shard ------
        print("[finalized] all shards spawned; polling for completion...", flush=True)
        deadline = time.time() + 12 * 3600  # 12 hr hard cap
        while True:
            all_done = True
            statuses = []
            for m in models:
                for fname in ("eval_call_ids_regular.json", "eval_call_ids_swap.json"):
                    d, r, e, total = stage_eval_status(run_ids[m.short_name], fname)
                    statuses.append(f"{m.short_name}/{fname.split('_')[-1].rstrip('.json')}: {d}/{total} (run={r}, err={e})")
                    if d + e < total:
                        all_done = False
            print(f"[finalized] {time.strftime('%H:%M:%S')}  " + "  ".join(statuses), flush=True)
            if all_done:
                break
            if time.time() > deadline:
                print("[finalized] hit 12hr deadline; stopping wait", flush=True)
                break
            time.sleep(180)

    # ------ Phase 5: aggregate per-model + per-corpus ------
    print("[finalized] aggregating per-model results...", flush=True)
    summaries = {}
    for m in models:
        run_id = run_ids[m.short_name]
        # Aggregate the regular shards (default subdir)
        s = stage_aggregate(m, run_id)
        summaries[m.short_name] = s
        # Also pull swap shards if present
        out_swap = _ensure_local_dir(f"{run_id}_results_swap")
        rc = _modal_volume_get(f"runs/{run_id}/shards_swap/", str(out_swap))
        if rc == 0:
            print(f"[finalized] swap shards synced -> {out_swap}", flush=True)

    summary_out = Path(f"local_data/finalized_summary_{args.run_tag}.json")
    summary_out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[finalized] all done; summary -> {summary_out}", flush=True)


if __name__ == "__main__":
    main()
