"""Subsample external eval corpora to 100 items for the 26B-A4B run.

The 26B model is ~3× slower per inference than E4B and runs on A100-80GB
($4-6/hr), so we use a smaller eval set to keep wall time and cost in
check.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 0
N_PER = 100

CONFIGS = [
    ("data/runs/moralchoice/eval.jsonl",
     "data/runs/moralchoice/eval_high_subset100.jsonl",
     lambda r: r.get("framing") == "high"),
    ("data/runs/ethics_util/eval.jsonl",
     "data/runs/ethics_util/eval_subset100.jsonl",
     lambda r: True),
    ("data/runs/ethics_deontology/eval.jsonl",
     "data/runs/ethics_deontology/eval_subset100.jsonl",
     lambda r: True),
]


def main():
    rng = random.Random(SEED)
    for src, dst, pred in CONFIGS:
        rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if pred(r)]
        rng.shuffle(rows)
        rows = rows[:N_PER]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{dst}: {len(rows)} items")


if __name__ == "__main__":
    main()
