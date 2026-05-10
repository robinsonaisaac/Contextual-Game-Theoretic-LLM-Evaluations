"""Spawn eval shards on E4B for MoralChoice + ETHICS Utilitarianism,
reusing the cooperation vector at runs/pd_full_v1/vectors.pt.

Each (corpus × alpha) pair becomes one shard. Output goes to a unique
result_subdir per corpus so shards don't collide.

Usage: python3 scripts/run_external_evals.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

RUN_ID = "pd_full_v1"
HF_ID = "google/gemma-4-E4B-it"
LAYER = 16
POSITION = "mean_trace"
ALPHAS = (-3.0, -1.0, 0.0, 1.0, 3.0)

CORPORA = [
    ("moralchoice", "shards_moralchoice", "data/runs/moralchoice/eval.jsonl"),
    ("ethics_util", "shards_ethics_util", "data/runs/ethics_util/eval.jsonl"),
]


def main():
    Cls = modal.Cls.from_name("safety", "SteeringWorker")
    worker = Cls(model_name=HF_ID)

    all_calls = {}
    for name, subdir, path in CORPORA:
        stories = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        print(f"[{name}] {len(stories)} stories", flush=True)
        calls = []
        for alpha in ALPHAS:
            fc = worker.eval_shard.spawn(
                run_id=RUN_ID,
                layer=LAYER,
                position=POSITION,
                alpha=float(alpha),
                stories=stories,
                result_subdir=subdir,
            )
            calls.append({"call_id": fc.object_id, "alpha": float(alpha)})
            print(f"  spawn alpha={alpha:+.1f} call_id={fc.object_id}", flush=True)
        all_calls[name] = {"subdir": subdir, "alphas": list(ALPHAS), "calls": calls}

    out = Path("local_data") / RUN_ID / "external_eval_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_calls, indent=2))
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
