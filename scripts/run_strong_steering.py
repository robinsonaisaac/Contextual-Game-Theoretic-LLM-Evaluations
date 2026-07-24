"""Spawn α=±5 (and ±6) shards on 26B-A4B for the two most informative
external benchmarks: MoralChoice high-ambig and ETHICS Utilitarianism.

We extend the existing dose-response curves at α=±3 to test whether the
behavioral effects amplify, saturate, or break down at stronger α.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

LARGE_RUN_ID = "pd_26B_A4B_v1"
LARGE_HF = "google/gemma-4-26B-A4B-it"
LARGE_LAYER = 11
POSITION = "mean_trace"
EXTRA_ALPHAS = (-6.0, -5.0, 5.0, 6.0)

CORPORA = [
    ("moralchoice_high", "shards_moralchoice_high",
     "data/runs/moralchoice/eval_high_subset100.jsonl"),
    ("ethics_util", "shards_ethics_util",
     "data/runs/ethics_util/eval_subset100.jsonl"),
    ("ethics_deontology", "shards_ethics_deontology",
     "data/runs/ethics_deontology/eval_subset100.jsonl"),
]


def main():
    Cls = modal.Cls.from_name("safety", "SteeringWorkerLarge")
    worker = Cls(model_name=LARGE_HF)

    all_calls = {}
    for name, subdir, path in CORPORA:
        stories = [json.loads(l) for l in
                   Path(path).read_text().splitlines() if l.strip()]
        print(f"\n[{name}] {len(stories)} stories")
        calls = []
        for alpha in EXTRA_ALPHAS:
            fc = worker.eval_shard.spawn(
                run_id=LARGE_RUN_ID, layer=LARGE_LAYER,
                position=POSITION, alpha=float(alpha),
                stories=stories, result_subdir=subdir,
            )
            calls.append({"call_id": fc.object_id, "alpha": float(alpha)})
            print(f"  α={alpha:+.1f} call_id={fc.object_id}")
        all_calls[name] = {"subdir": subdir, "calls": calls}

    out = Path("local_data") / "strong_steering_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_calls, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
