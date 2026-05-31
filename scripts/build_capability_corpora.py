"""Build GSM8k (math reasoning) + HumanEval (coding) corpora for the steering
capability-regression ablation.

These mirror build_reasoning_corpora.py but score differently (see
game_theory_llm.capability_scoring): GSM8k is free-form numeric, HumanEval is
execution-based pass@1. The steering eval shard only persists the generation
``trace`` (plus story_id), so we keep the gold answer / unit tests in this
local JSONL and join on story_id at scoring time.

``coop_choice`` is a dummy ("A") that the on-device parser will mark wrong; we
ignore the on-device score and re-score traces locally.

Usage: python3 scripts/build_capability_corpora.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from game_theory_llm.capability_scoring import gold_gsm8k_answer

SEED = 11
N_GSM8K = 250          # held-out math items
GSM8K_MAX_NEW = 1024   # room for chain-of-thought
HUMANEVAL_MAX_NEW = 1024

GSM8K_PROMPT = """{question}

Solve this step by step. Show your reasoning, then give your final numerical answer in the form <answer>NUMBER</answer>."""

HUMANEVAL_PROMPT = """Complete the following Python function. Return the COMPLETE function (signature, body, and any imports) inside a single ```python ...``` code block. Do not include explanation or tests.

```python
{prompt}```"""


def build_gsm8k(rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:N_GSM8K]
    out = []
    for i in idxs:
        ex = ds[i]
        out.append({
            "story_id": f"gsm8k_{i:05d}",
            "prompt": GSM8K_PROMPT.format(question=ex["question"].strip()),
            "coop_choice": "A",                       # dummy; ignored
            "gsm8k_gold": gold_gsm8k_answer(ex["answer"]),
            "seed": rng.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "max_new_tokens": GSM8K_MAX_NEW,
            "game_type": "gsm8k",
            "framing": "math_reasoning",
        })
    return out


def build_humaneval(rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for ex in ds:
        out.append({
            "story_id": ex["task_id"].replace("/", "_"),   # e.g. HumanEval_0
            "prompt": HUMANEVAL_PROMPT.format(prompt=ex["prompt"]),
            "coop_choice": "A",                       # dummy; ignored
            "he_prompt": ex["prompt"],                # signature+docstring stub
            "he_test": ex["test"],                    # defines check(candidate)
            "he_entry_point": ex["entry_point"],
            "seed": rng.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "max_new_tokens": HUMANEVAL_MAX_NEW,
            "game_type": "humaneval",
            "framing": "code_generation",
        })
    return out


def main():
    rng = random.Random(SEED)
    for name, builder in [("gsm8k", build_gsm8k), ("humaneval", build_humaneval)]:
        print(f"\n[{name}] building...")
        rows = builder(rng)
        out = Path(f"data/runs/capability/{name}_eval.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {len(rows)} items -> {out}")


if __name__ == "__main__":
    main()
