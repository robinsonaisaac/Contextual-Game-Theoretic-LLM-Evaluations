"""Spawn follow-up eval shards:
  (1) E4B + ETHICS-Deontology (200 items)   — uses pd_full_v1 vectors
  (2) 26B-A4B + ETHICS-Deontology (100 items) — uses pd_26B_A4B_v1 vectors L11
  (3) 26B-A4B + ETHICS-Utilitarianism (100 items) — uses pd_26B_A4B_v1 L11
  (4) 26B-A4B + MoralChoice high-ambig (100 items) — uses pd_26B_A4B_v1 L11

The 26B-A4B model uses SteeringWorkerLarge (A100-80GB) and L11.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ALPHAS = (-3.0, -1.0, 0.0, 1.0, 3.0)
POSITION = "mean_trace"

E4B_RUN_ID = "pd_full_v1"
E4B_HF = "google/gemma-4-E4B-it"
E4B_LAYER = 16

LARGE_RUN_ID = "pd_26B_A4B_v1"
LARGE_HF = "google/gemma-4-26B-A4B-it"
LARGE_LAYER = 11


def spawn(worker, run_id, layer, position, stories, subdir):
    calls = []
    for alpha in ALPHAS:
        fc = worker.eval_shard.spawn(
            run_id=run_id, layer=layer, position=position,
            alpha=float(alpha), stories=stories, result_subdir=subdir,
        )
        calls.append({"call_id": fc.object_id, "alpha": float(alpha)})
        print(f"    α={alpha:+.1f} call_id={fc.object_id}", flush=True)
    return calls


def main():
    SmallCls = modal.Cls.from_name("safety", "SteeringWorker")
    LargeCls = modal.Cls.from_name("safety", "SteeringWorkerLarge")
    e4b = SmallCls(model_name=E4B_HF)
    large = LargeCls(model_name=LARGE_HF)

    all_calls: dict[str, dict] = {}

    # (1) E4B + Deontology (200 items)
    stories = [json.loads(l) for l in
               Path("data/runs/ethics_deontology/eval.jsonl").read_text().splitlines() if l.strip()]
    print(f"\n[E4B + Deontology] {len(stories)} items")
    all_calls["e4b_deontology"] = {
        "model": "E4B", "run_id": E4B_RUN_ID, "layer": E4B_LAYER,
        "subdir": "shards_ethics_deontology",
        "calls": spawn(e4b, E4B_RUN_ID, E4B_LAYER, POSITION, stories, "shards_ethics_deontology"),
    }

    # (2) 26B + Deontology (100 items)
    stories = [json.loads(l) for l in
               Path("data/runs/ethics_deontology/eval_subset100.jsonl").read_text().splitlines() if l.strip()]
    print(f"\n[26B + Deontology] {len(stories)} items")
    all_calls["26b_deontology"] = {
        "model": "26B-A4B", "run_id": LARGE_RUN_ID, "layer": LARGE_LAYER,
        "subdir": "shards_ethics_deontology",
        "calls": spawn(large, LARGE_RUN_ID, LARGE_LAYER, POSITION, stories, "shards_ethics_deontology"),
    }

    # (3) 26B + Utilitarianism (100 items)
    stories = [json.loads(l) for l in
               Path("data/runs/ethics_util/eval_subset100.jsonl").read_text().splitlines() if l.strip()]
    print(f"\n[26B + Util] {len(stories)} items")
    all_calls["26b_util"] = {
        "model": "26B-A4B", "run_id": LARGE_RUN_ID, "layer": LARGE_LAYER,
        "subdir": "shards_ethics_util",
        "calls": spawn(large, LARGE_RUN_ID, LARGE_LAYER, POSITION, stories, "shards_ethics_util"),
    }

    # (4) 26B + MoralChoice high-ambig (100 items)
    stories = [json.loads(l) for l in
               Path("data/runs/moralchoice/eval_high_subset100.jsonl").read_text().splitlines() if l.strip()]
    print(f"\n[26B + MoralChoice high-ambig] {len(stories)} items")
    all_calls["26b_moralchoice_high"] = {
        "model": "26B-A4B", "run_id": LARGE_RUN_ID, "layer": LARGE_LAYER,
        "subdir": "shards_moralchoice_high",
        "calls": spawn(large, LARGE_RUN_ID, LARGE_LAYER, POSITION, stories, "shards_moralchoice_high"),
    }

    out = Path("local_data") / "followup_eval_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_calls, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
