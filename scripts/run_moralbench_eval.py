"""Spawn MoralBench eval shards on E4B at layer 16, reusing the
already-fitted cooperation vector from `pd_full_v1`. Writes shards to
`runs/pd_full_v1/shards_moralbench/`.

Usage:
    python3 scripts/run_moralbench_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

RUN_ID = "pd_full_v1"      # cooperation vector is here
HF_ID = "google/gemma-4-E4B-it"
LAYER = 16
POSITION = "mean_trace"
ALPHAS = (-3.0, -1.0, 0.0, 1.0, 3.0)
SUBDIR = "shards_moralbench"
EVAL_PATH = Path("data/runs/moralbench/eval.jsonl")


def main():
    stories = [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]
    print(f"[eval] {len(stories)} MoralBench items", flush=True)

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

    out = Path("local_data") / RUN_ID / "moralbench_eval_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": RUN_ID, "subdir": SUBDIR, "layer": LAYER, "position": POSITION,
        "alphas": list(ALPHAS), "n_stories": len(stories), "calls": calls,
    }, indent=2))
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
