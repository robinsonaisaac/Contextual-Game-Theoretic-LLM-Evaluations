"""Run powered (N=300) eval shards on E4B at α ∈ {-6, -3, 0, +3, +6} for
MoralChoice high-ambig and ETHICS-Util. We use E4B (not 26B) to keep cost
manageable at 5× the eval N.

Output goes to subdirs `shards_moralchoice_high_p300` and
`shards_ethics_util_p300` under runs/pd_full_v1/ to avoid colliding with
the existing N=100/200 shard dirs.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

RUN_ID = "pd_full_v1"
HF_ID = "google/gemma-4-E4B-it"
LAYER = 16
POSITION = "mean_trace"
ALPHAS = (-6.0, -3.0, 0.0, 3.0, 6.0)

CORPORA = [
    ("moralchoice_high_p300", "shards_moralchoice_high_p300",
     "data/runs/moralchoice/eval_high_powered300.jsonl"),
    ("ethics_util_p300", "shards_ethics_util_p300",
     "data/runs/ethics_util/eval_powered300.jsonl"),
]


def main():
    Cls = modal.Cls.from_name("safety", "SteeringWorker")
    worker = Cls(model_name=HF_ID)

    all_calls = {}
    for name, subdir, path in CORPORA:
        stories = [json.loads(l) for l in
                   Path(path).read_text().splitlines() if l.strip()]
        print(f"\n[{name}] {len(stories)} stories")
        calls = []
        for alpha in ALPHAS:
            fc = worker.eval_shard.spawn(
                run_id=RUN_ID, layer=LAYER, position=POSITION,
                alpha=float(alpha), stories=stories, result_subdir=subdir,
            )
            calls.append({"call_id": fc.object_id, "alpha": float(alpha)})
            print(f"  α={alpha:+.1f} call_id={fc.object_id}")
        all_calls[name] = {"subdir": subdir, "calls": calls}

    out = Path("local_data") / "powered_evals_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_calls, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
