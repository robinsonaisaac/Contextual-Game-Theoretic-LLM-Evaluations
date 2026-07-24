"""Spawn rationality-eval shards on E4B at layer 16 only, with a 5-alpha
grid. Detached spawn — caller polls + aggregates separately.

Usage:
    python3 scripts/run_rationality_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

RUN_ID = "rat_E4B_v1"
HF_ID = "google/gemma-4-E4B-it"
LAYER = 16
POSITION = "mean_trace"
ALPHAS = (-3.0, -1.0, 0.0, 1.0, 3.0)
SUBDIR = "shards"
EVAL_PATH = Path("data/runs/2026-05-05-sharp/rationality/heldout_eval_subset.jsonl")


def main():
    stories = [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]
    print(f"[eval] {len(stories)} held-out stories", flush=True)

    Cls = modal.Cls.from_name("safety", "SteeringWorker")
    worker = Cls(model_name=HF_ID)

    calls = []
    for alpha in ALPHAS:
        fc = worker.eval_shard.spawn(
            run_id=RUN_ID,
            layer=LAYER,
            position=POSITION,
            alpha=float(alpha),
            stories=stories,
            result_subdir=SUBDIR,
        )
        calls.append({
            "call_id": fc.object_id,
            "layer": LAYER,
            "position": POSITION,
            "alpha": float(alpha),
        })
        print(f"[spawn] layer={LAYER} alpha={alpha:+.1f} call_id={fc.object_id}", flush=True)

    out = Path("local_data") / RUN_ID / "rationality_eval_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": RUN_ID, "layer": LAYER, "position": POSITION,
        "alphas": list(ALPHAS), "n_stories": len(stories), "calls": calls,
    }, indent=2))
    print(f"[saved] {out}", flush=True)
    print(f"[next] poll with: python3 scripts/run_rationality_eval.py --poll", flush=True)


if __name__ == "__main__":
    main()
