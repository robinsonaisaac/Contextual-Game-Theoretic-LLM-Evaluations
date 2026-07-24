#!/usr/bin/env python3
"""End-to-end steering pipeline driver — single entrypoint for all models.

The pipeline has six stages, all idempotent:

  1. download     Snapshot model weights into the safety Modal volume.
  2. warmup       One-shot architecture discovery on the GPU; sanity check.
  3. extract      Run train-set extract: generate trace + capture activations
                  for every story. Writes per-story bundles to the volume.
  4. fit          Sync bundles locally, mean-diff-fit steering vectors,
                  push vectors.pt back to the volume.
  5. eval         Spawn N (layer, alpha) shards detached on Modal. Each
                  writes its own results parquet + appends to a shared
                  progress JSONL.
  6. aggregate    Pull all shard parquets, stitch into one results table,
                  print summary stats.

Usage:
  python3 scripts/run_full_pipeline.py --model E4B --stage all
  python3 scripts/run_full_pipeline.py --model E2B --stage extract
  python3 scripts/run_full_pipeline.py --model 26B-A4B --stage eval --wait
  python3 scripts/run_full_pipeline.py --model all --stage all
  python3 scripts/run_full_pipeline.py --model all --stage extract --parallel

State on the volume per model lives at:
  /data/runs/{run_id}/                  -- run_id = pd_{NAME}_v1
      activations/train/*.pt            -- one bundle per story
      index.parquet                     -- metadata index
      vectors.pt                        -- fitted SteeringVectorSet
      shards/*.parquet                  -- per-cell eval results
      eval_progress.jsonl               -- live progress log

Stories file:
  data/runs/2026-05-05-sharp/steering/full_corpus.jsonl   -- 324 PD stories,
      already chat-templatable; coop_choice="A" for every story.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from game_theory_llm.steering.configs import (
    MODEL_CONFIGS,
    ModelConfig,
    get_config,
    run_id_for,
)

# --- Defaults ---------------------------------------------------------------

DEFAULT_TRAIN_STORIES = "data/runs/2026-05-05-sharp/steering/train.jsonl"
DEFAULT_EVAL_STORIES = "data/runs/2026-05-05-sharp/steering/full_corpus.jsonl"
DEFAULT_ALPHA_GRID = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
APP_NAME = "safety"
VOLUME_NAME = "safety"


# --- Helpers ----------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_local_dir(name: str) -> Path:
    p = Path("local_data") / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _modal_volume_get(remote: str, local: str) -> int:
    """Return code from `modal volume get`."""
    return subprocess.call(
        ["python3", "-m", "modal", "volume", "get", "--force",
         VOLUME_NAME, remote, local],
        stderr=subprocess.DEVNULL,
    )


def _modal_volume_put(local: str, remote: str) -> None:
    subprocess.check_call([
        "python3", "-m", "modal", "volume", "put", "--force",
        VOLUME_NAME, local, remote,
    ])


# --- Stage implementations --------------------------------------------------

def stage_download(cfg: ModelConfig) -> dict:
    """Pull weights into the volume so worker containers don't fetch HF."""
    import modal
    from game_theory_llm.steering import modal_app  # noqa: F401 ensure deploy
    fn = modal.Function.from_name(APP_NAME, "download_model_once")
    print(f"[download] {cfg.hf_id} -> volume...", flush=True)
    result = fn.remote(cfg.hf_id)
    print(f"[download] {result}", flush=True)
    return result


def stage_warmup(cfg: ModelConfig) -> dict:
    import modal
    fn_name = "warmup" if cfg.gpu_tier == "small" else "warmup_large"
    fn = modal.Function.from_name(APP_NAME, fn_name)
    print(f"[warmup] running {fn_name}({cfg.hf_id})...", flush=True)
    result = fn.remote(cfg.hf_id)
    expected_layers = cfg.n_layers
    actual = result.get("n_layers")
    if actual != expected_layers:
        print(f"[warmup] WARNING: configs.py says n_layers={expected_layers} "
              f"but warmup found {actual}", flush=True)
    print(f"[warmup] n_layers={actual} hidden_dim={result.get('hidden_dim')} "
          f"hook_ok={result.get('hook_ok')}", flush=True)
    return result


def stage_extract(cfg: ModelConfig, run_id: str, stories_path: Path) -> str:
    """Spawn a detached extract for the train corpus. Returns call_id."""
    import modal
    cls_name = "SteeringWorker" if cfg.gpu_tier == "small" else "SteeringWorkerLarge"
    Cls = modal.Cls.from_name(APP_NAME, cls_name)
    worker = Cls(model_name=cfg.hf_id)
    stories = _read_jsonl(stories_path)
    print(f"[extract] {cfg.short_name}: dispatching {len(stories)} stories "
          f"-> run_id={run_id}", flush=True)
    fc = worker.extract.spawn(stories, run_id, "train")
    call_id_file = _ensure_local_dir(run_id) / "extract_call_id.txt"
    call_id_file.write_text(fc.object_id)
    print(f"[extract] call_id={fc.object_id} (saved to {call_id_file})", flush=True)
    return fc.object_id


def stage_extract_wait(call_id: str) -> dict:
    """Block until an extract spawn completes; return the summary."""
    import modal
    fc = modal.FunctionCall.from_id(call_id)
    print(f"[extract-wait] blocking on {call_id}...", flush=True)
    result = fc.get()
    print(f"[extract-wait] done: coop_rate={result.get('coop_rate'):.3f} "
          f"n={result.get('n_stories')}", flush=True)
    return result


def stage_fit(cfg: ModelConfig, run_id: str) -> dict:
    """Sync bundles, fit vectors locally, push vectors.pt back."""
    from game_theory_llm.steering.storage import (
        load_activation_bundle, read_index, save_vector_set,
    )
    from game_theory_llm.steering.vector_fitting import fit_vectors

    local_root = _ensure_local_dir(f"{run_id}_dl")
    rc = _modal_volume_get(f"runs/{run_id}/", str(local_root))
    if rc != 0:
        sys.exit(f"[fit] failed to sync runs/{run_id} (rc={rc})")

    local_run_dir = local_root / run_id
    if not local_run_dir.exists():
        # Some Modal CLI versions strip the run_id segment; fall back.
        local_run_dir = local_root
    index_path = local_run_dir / "index.parquet"
    if not index_path.exists():
        sys.exit(f"[fit] index.parquet not at {index_path}; check sync")

    df = read_index(index_path, split="train")
    bundles = []
    for row in df.itertuples(index=False):
        rel = Path(row.path).relative_to(f"/data/runs/{run_id}")
        bundles.append(load_activation_bundle(local_run_dir / rel))

    print(f"[fit] loaded {len(bundles)} bundles for {cfg.short_name}", flush=True)
    vs = fit_vectors(bundles, model_name=cfg.short_name.lower())
    out_path = local_run_dir / "vectors.pt"
    save_vector_set(vs, out_path)

    # Push back to volume
    _modal_volume_put(str(out_path), f"runs/{run_id}/vectors.pt")
    summary = {
        "run_id": run_id,
        "n_vectors": len(vs.vectors),
        "n_bundles": len(bundles),
        "corpus_hash": vs.corpus_hash,
        "vectors_path": str(out_path),
    }
    print(f"[fit] {summary}", flush=True)
    return summary


def stage_eval(cfg: ModelConfig, run_id: str, stories_path: Path,
               alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
               positions: tuple[str, ...] = ("mean_trace",),
               result_subdir: str = "shards",
               multi_result_subdir: str = "shards_multi",
               call_ids_filename: str = "eval_call_ids.json") -> dict:
    """Spawn a (layer × alpha) eval grid as N detached shards.

    `result_subdir`/`multi_result_subdir` let callers route swap-corpus
    shards to e.g. ``shards_swap``/``shards_multi_swap`` so they don't
    overwrite the regular run's parquets.
    """
    import modal
    cls_name = "SteeringWorker" if cfg.gpu_tier == "small" else "SteeringWorkerLarge"
    Cls = modal.Cls.from_name(APP_NAME, cls_name)
    worker = Cls(model_name=cfg.hf_id)

    stories = _read_jsonl(stories_path)
    shards = []
    for layer in cfg.candidate_layers:
        for pos in positions:
            for alpha in alpha_grid:
                fc = worker.eval_shard.spawn(
                    run_id=run_id,
                    layer=layer,
                    position=pos,
                    alpha=float(alpha),
                    stories=stories,
                    result_subdir=result_subdir,
                )
                shards.append({
                    "kind": "single", "call_id": fc.object_id,
                    "layer": layer, "position": pos, "alpha": float(alpha),
                    "result_subdir": result_subdir,
                })
    for cells in cfg.multi_cells:
        cells_list = [list(c) for c in cells]
        label = "+".join(f"L{l}_{p}" for l, p in cells)
        for alpha in alpha_grid:
            fc = worker.eval_shard_multi.spawn(
                run_id=run_id,
                cells=cells_list,
                alpha=float(alpha),
                stories=stories,
                label=label,
                result_subdir=multi_result_subdir,
            )
            shards.append({
                "kind": "multi", "call_id": fc.object_id,
                "cells": cells_list, "label": label, "alpha": float(alpha),
                "result_subdir": multi_result_subdir,
            })

    out_path = _ensure_local_dir(run_id) / call_ids_filename
    out_path.write_text(json.dumps({
        "run_id": run_id,
        "model": cfg.short_name,
        "n_shards": len(shards),
        "n_stories": len(stories),
        "alpha_grid": list(alpha_grid),
        "positions": list(positions),
        "shards": shards,
    }, indent=2))
    print(f"[eval] {cfg.short_name}: {len(shards)} shards spawned -> {out_path}", flush=True)
    return {"call_ids_file": str(out_path), "n_shards": len(shards)}


def stage_eval_status(run_id: str, call_ids_filename: str = "eval_call_ids.json") -> tuple[int, int, int, int]:
    """Return (done, running, errored, total) for an eval call_ids set."""
    import modal
    blob = json.loads((Path("local_data") / run_id / call_ids_filename).read_text())
    n_done = n_running = n_err = 0
    for sh in blob["shards"]:
        fc = modal.FunctionCall.from_id(sh["call_id"])
        try:
            fc.get(timeout=0)
            n_done += 1
        except TimeoutError:
            n_running += 1
        except Exception:
            n_err += 1
    return n_done, n_running, n_err, len(blob["shards"])


def stage_eval_wait(run_id: str, poll_seconds: int = 120,
                    timeout_seconds: int = 36000,
                    call_ids_filename: str = "eval_call_ids.json") -> None:
    """Block until all shards in a spawn finish (or hit a hard timeout)."""
    deadline = time.time() + timeout_seconds
    while True:
        d, r, e, total = stage_eval_status(run_id, call_ids_filename=call_ids_filename)
        print(f"[eval-wait] {run_id} ({call_ids_filename}): done={d}/{total} running={r} err={e}", flush=True)
        if d + e >= total:
            return
        if time.time() > deadline:
            print(f"[eval-wait] hit timeout {timeout_seconds}s", flush=True)
            return
        time.sleep(poll_seconds)


def stage_aggregate(cfg: ModelConfig, run_id: str) -> dict:
    """Pull every per-shard parquet from the volume into one dataframe."""
    import pandas as pd

    out_root = _ensure_local_dir(f"{run_id}_results")
    rc = _modal_volume_get(f"runs/{run_id}/shards/", str(out_root))
    pqs = sorted((out_root).rglob("*.parquet"))
    if not pqs:
        msg = (f"no shard parquets at runs/{run_id}/shards on the volume. "
               f"Did the eval stage run yet?")
        print(f"[aggregate] {msg}", flush=True)
        return {"error": msg, "rc": rc}

    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    df["parsed"] = df.decision.fillna("").astype(str).str.len() > 0
    df.to_parquet(out_root / "results_full.parquet", index=False)

    # Summary
    summary_rows = []
    for (layer, pos), g in df.groupby(["layer", "position"]):
        for alpha in sorted(g.alpha.unique()):
            sub = g[(g.alpha == alpha) & g.parsed]
            n = len(sub); c = int(sub.cooperated.sum())
            summary_rows.append({
                "layer": layer, "position": pos, "alpha": alpha,
                "n_parsed": n, "n_cooperated": c,
                "coop_rate": c / max(1, n),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(out_root / "summary.parquet", index=False)

    # Print best-cell per layer
    print(f"\n[aggregate] {cfg.short_name} ({run_id}):")
    print(summary.pivot_table(
        index="layer", columns="alpha", values="coop_rate",
    ).round(3).to_string())
    print(f"\n[aggregate] wrote {out_root}/results_full.parquet "
          f"({len(df)} rows) and summary.parquet")

    return {"results_full": str(out_root / "results_full.parquet"),
            "summary": str(out_root / "summary.parquet"),
            "n_rows": len(df)}


# --- Stage dispatcher -------------------------------------------------------

def run_for_model(cfg: ModelConfig, stage: str, *,
                  train_stories: Path,
                  eval_stories: Path,
                  alpha_grid: tuple[float, ...],
                  wait: bool,
                  run_id_override: str | None = None) -> None:
    run_id = run_id_override or run_id_for(cfg.short_name)
    print(f"\n=== {cfg.short_name} ({cfg.hf_id})  run_id={run_id}  stage={stage} ===")

    if stage in ("download", "all"):
        stage_download(cfg)

    if stage in ("warmup", "all"):
        stage_warmup(cfg)

    if stage in ("extract", "all"):
        call_id = stage_extract(cfg, run_id, train_stories)
        if wait or stage == "all":
            stage_extract_wait(call_id)

    if stage in ("fit", "all"):
        stage_fit(cfg, run_id)

    if stage in ("eval", "all"):
        stage_eval(cfg, run_id, eval_stories, alpha_grid=alpha_grid)
        if wait:
            stage_eval_wait(run_id)

    if stage in ("aggregate", "all"):
        stage_aggregate(cfg, run_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Model short_name from configs.py (E2B, E4B, 26B-A4B), "
                         "or 'all' to iterate every config.")
    ap.add_argument("--stage", required=True,
                    choices=["download", "warmup", "extract", "fit", "eval",
                             "aggregate", "all"])
    ap.add_argument("--train-stories", default=DEFAULT_TRAIN_STORIES)
    ap.add_argument("--eval-stories", default=DEFAULT_EVAL_STORIES)
    ap.add_argument("--alpha-grid", default=",".join(str(a) for a in DEFAULT_ALPHA_GRID))
    ap.add_argument("--wait", action="store_true",
                    help="Block until eval shards complete before returning.")
    ap.add_argument("--run-id",
                    help="Override the auto-derived run_id (e.g. for legacy data).")
    args = ap.parse_args()

    alpha_grid = tuple(float(x) for x in args.alpha_grid.split(","))
    train_stories = Path(args.train_stories)
    eval_stories = Path(args.eval_stories)

    if args.model == "all":
        targets = list(MODEL_CONFIGS.values())
    else:
        targets = [get_config(args.model)]

    if args.run_id and args.model == "all":
        sys.exit("--run-id only makes sense with a single --model")

    for cfg in targets:
        run_for_model(cfg, args.stage,
                      train_stories=train_stories,
                      eval_stories=eval_stories,
                      alpha_grid=alpha_grid,
                      wait=args.wait,
                      run_id_override=args.run_id)


if __name__ == "__main__":
    main()
