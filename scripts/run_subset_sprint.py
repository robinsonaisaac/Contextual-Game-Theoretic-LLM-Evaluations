#!/usr/bin/env python3
"""Reproducible 'sprint' driver: full multi-model + swap diagnostic on a
small fixed subset of the steering corpus, using already-fitted vectors.

This is the script that produced the numbers in the paper section's
unified table when run on a 50-story random subset (seed 0). Each shard
processes 50 stories instead of the full 324, which lets every shard
finish in ~10-15 minutes wall clock while still being statistically
informative for the negative-direction effect (where the model's
behavior is large).

Prerequisites (run once each):
    1. python3 -m modal deploy game_theory_llm/steering/modal_app.py
    2. python3 scripts/build_subset_corpus.py \\
           --regular .../full_corpus.jsonl --swap .../full_corpus_swap.jsonl \\
           --out-regular .../full_corpus_subset50.jsonl \\
           --out-swap    .../full_corpus_swap_subset50.jsonl
    3. Vectors must already be on the volume per model. If not, run
       run_full_pipeline.py --stage extract + --stage fit per model first.

Usage:
    python3 scripts/run_subset_sprint.py
    python3 scripts/run_subset_sprint.py --no-wait    # spawn only
    python3 scripts/run_subset_sprint.py --aggregate  # poll + summarise

Layout per shard on the safety volume:
    /data/runs/{run_id}/shards_subset50/L{layer}_{pos}_a{alpha}.parquet
    /data/runs/{run_id}/shards_subset50_swap/{layer}_{pos}_a{alpha}.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import modal


# Per-model spec: (run_id with vectors.pt, hf_id, gpu_tier, candidate_layer)
DEFAULT_MODEL_SPEC = [
    ("pd_E2B_v1",     "google/gemma-4-E2B-it",     "small", 17),
    ("pd_full_v1",    "google/gemma-4-E4B-it",     "small", 16),
    ("pd_26B_A4B_v1", "google/gemma-4-26B-A4B-it", "large", 11),
]
DEFAULT_ALPHAS = (-3.0, -1.0, 0.0, 1.0, 3.0)
APP = "safety"
SUBDIR_REG = "shards_subset50"
SUBDIR_SWAP = "shards_subset50_swap"


def _spawn_all(spec, alphas, regular, swap):
    calls = []
    for run_id, hf, tier, layer in spec:
        cls = "SteeringWorker" if tier == "small" else "SteeringWorkerLarge"
        Cls = modal.Cls.from_name(APP, cls)
        worker = Cls(model_name=hf)
        for label, stories, subdir in [
            ("reg", regular, SUBDIR_REG),
            ("swap", swap, SUBDIR_SWAP),
        ]:
            for alpha in alphas:
                fc = worker.eval_shard.spawn(
                    run_id=run_id, layer=layer, position="mean_trace",
                    alpha=float(alpha), stories=stories, result_subdir=subdir,
                )
                calls.append({
                    "run_id": run_id, "model": hf, "tier": tier,
                    "layer": layer, "alpha": float(alpha),
                    "corpus": label, "subdir": subdir,
                    "call_id": fc.object_id,
                })
    return calls


def _poll_until_done(calls, poll_seconds=120, timeout_seconds=3600):
    deadline = time.time() + timeout_seconds
    while True:
        n_done = n_run = n_err = 0
        for c in calls:
            fc = modal.FunctionCall.from_id(c["call_id"])
            try:
                fc.get(timeout=0); n_done += 1
            except TimeoutError:
                n_run += 1
            except Exception:
                n_err += 1
        print(f"[sprint] done={n_done}/{len(calls)} running={n_run} err={n_err}", flush=True)
        if n_done + n_err == len(calls):
            return
        if time.time() > deadline:
            print(f"[sprint] hit {timeout_seconds}s timeout; stopping", flush=True)
            return
        time.sleep(poll_seconds)


def _summarise(calls):
    rows = []
    for c in calls:
        fc = modal.FunctionCall.from_id(c["call_id"])
        try:
            r = fc.get(timeout=0)
        except Exception:
            continue
        rows.append({
            "model": c["model"].split("/")[-1],
            "layer": c["layer"], "alpha": c["alpha"],
            "corpus": c["corpus"],
            "coop_rate": r.get("cooperation_rate"),
            "n": r.get("n_stories"),
        })
    rows.sort(key=lambda r: (r["model"], r["corpus"], r["alpha"]))
    print()
    print(f"{'model':<25}{'corpus':<6}{'α':>5}  {'coop':>5}")
    print("-" * 50)
    for r in rows:
        print(f"{r['model']:<25}{r['corpus']:<6}{r['alpha']:>+.1f}  {r['coop_rate']:.3f}")
    return rows


def _decompose(rows):
    """Cooperation vs A-letter-bias via the additive decomposition."""
    by_key = {(r["model"], r["alpha"], r["corpus"]): r for r in rows}
    print()
    print("Cooperation/A-bias decomposition (parsed-only)")
    print(f"{'model':<25}{'α':>5}  {'reg':>5}  {'swap':>5}  {'e_coop':>7}  {'e_A':>6}")
    print("-" * 65)
    for model in sorted({r["model"] for r in rows}):
        # need α=0 from both corpora as baselines
        b_reg = by_key.get((model, 0.0, "reg"))
        b_swap = by_key.get((model, 0.0, "swap"))
        if b_reg is None or b_swap is None:
            continue
        m_base = (b_reg["coop_rate"] + b_swap["coop_rate"]) / 2
        d_base = (b_reg["coop_rate"] - b_swap["coop_rate"]) / 2
        for alpha in sorted({r["alpha"] for r in rows if r["model"] == model}):
            r_reg = by_key.get((model, alpha, "reg"))
            r_swap = by_key.get((model, alpha, "swap"))
            if r_reg is None or r_swap is None:
                continue
            mean = (r_reg["coop_rate"] + r_swap["coop_rate"]) / 2
            half_diff = (r_reg["coop_rate"] - r_swap["coop_rate"]) / 2
            e_coop = mean - m_base
            e_A = half_diff - d_base
            print(f"{model:<25}{alpha:>+.1f}  "
                  f"{r_reg['coop_rate']:.3f}  {r_swap['coop_rate']:.3f}  "
                  f"{e_coop:>+7.3f}  {e_A:>+6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular-stories",
                    default="data/runs/2026-05-05-sharp/steering/full_corpus_subset50.jsonl")
    ap.add_argument("--swap-stories",
                    default="data/runs/2026-05-05-sharp/steering/full_corpus_swap_subset50.jsonl")
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--calls-out", default="local_data/subset_sprint_calls.json")
    ap.add_argument("--no-wait", action="store_true",
                    help="Spawn shards but don't poll for completion.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Skip spawning; poll an existing calls file and aggregate.")
    args = ap.parse_args()

    calls_path = Path(args.calls_out)

    if not args.aggregate:
        regular = [json.loads(l) for l in Path(args.regular_stories).read_text().splitlines() if l.strip()]
        swap = [json.loads(l) for l in Path(args.swap_stories).read_text().splitlines() if l.strip()]
        if len(regular) != len(swap):
            sys.exit(f"corpora misaligned: {len(regular)} reg vs {len(swap)} swap")
        alphas = tuple(float(x) for x in args.alphas.split(","))

        calls = _spawn_all(DEFAULT_MODEL_SPEC, alphas, regular, swap)
        calls_path.parent.mkdir(parents=True, exist_ok=True)
        calls_path.write_text(json.dumps(calls, indent=2))
        print(f"[sprint] spawned {len(calls)} shards -> {calls_path}", flush=True)

        if args.no_wait:
            return

    if not calls_path.exists():
        sys.exit(f"no calls file at {calls_path}; run without --aggregate first")

    calls = json.loads(calls_path.read_text())
    _poll_until_done(calls)
    rows = _summarise(calls)
    _decompose(rows)


if __name__ == "__main__":
    main()
