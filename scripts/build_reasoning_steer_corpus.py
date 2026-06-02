"""Build a GSM8k TRAIN corpus for fitting a 'correctness' steering vector.

The steering pipeline generates a solution per problem, labels it correct vs
incorrect (capability_scoring.gsm8k_correct, wired into _impl_extract), and
mean-difference-fits a direction = mean(correct activations) - mean(incorrect).
Steering along +alpha then tests whether pushing toward 'correct-like' states
improves reasoning. Train items come from the GSM8k TRAIN split, disjoint from
the test-split eval corpus (data/runs/capability/gsm8k_eval.jsonl).

Usage: python3 scripts/build_reasoning_steer_corpus.py [--n 300]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from game_theory_llm.capability_scoring import gold_gsm8k_answer

SEED = 7
PROMPT = """{question}

Solve this step by step. Show your reasoning, then give your final numerical answer in the form <answer>NUMBER</answer>."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/runs/reason_steer/gsm8k_train.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in idxs:
            ex = ds[i]
            f.write(json.dumps({
                "story_id": f"gsm8ktrain_{i:05d}",
                "prompt": PROMPT.format(question=ex["question"].strip()),
                "coop_choice": "A",                  # dummy; correctness is the label
                "gsm8k_gold": gold_gsm8k_answer(ex["answer"]),
                "seed": rng.randint(0, 2**16 - 1),
                "temperature": 0.7,
                "max_new_tokens": 384,
                "game_type": "gsm8k",
                "framing": "math_reasoning",
            }) + "\n")
    print(f"{len(idxs)} train items -> {out}")


if __name__ == "__main__":
    main()
