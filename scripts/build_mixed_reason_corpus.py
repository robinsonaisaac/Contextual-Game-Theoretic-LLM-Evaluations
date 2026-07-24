"""Build a MIXED reasoning corpus (GSM8k + MMLU) for fitting a *general*
correctness steering vector.

Motivation: a vector fit on GSM8k alone gives a big in-domain lift but does not
generalize (it mildly hurts MMLU/GPQA accuracy-among-parsed). Fitting the
correctness direction across two distinct reasoning tasks tests whether a more
task-general "get-it-right" direction exists.

Both tasks are labelled by correctness in _impl_extract:
  - GSM8k via numeric match (game_type == "gsm8k", uses gsm8k_gold);
  - MMLU via decision == answer (game_type == "mmlu", uses coop_choice).
Train items are disjoint from the eval corpora:
  - GSM8k: train split (reuses gsm8k_train.jsonl, built from GSM8k-train);
  - MMLU: items [300:300+N] of the seed-11 test shuffle (eval uses [0:300]).

Usage: python3 scripts/build_mixed_reason_corpus.py [--n-mmlu 150 --n-gsm8k 150]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPT_4 = """{question}

A. {a}
B. {b}
C. {c}
D. {d}

Reason briefly, then end with <decision>A</decision>, <decision>B</decision>, <decision>C</decision>, or <decision>D</decision>."""


def build_mmlu_offset(n: int, skip: int = 300):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(11)                  # same seed as build_reasoning_corpora
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[skip:skip + n * 2]           # past the eval slice; *2 for 4-option filter
    rng2 = random.Random(123)
    out = []
    for i in idxs:
        ex = ds[i]
        if len(ex["choices"]) != 4:
            continue
        out.append({
            "story_id": f"mmlutrain_{i:05d}",
            "prompt": PROMPT_4.format(question=ex["question"].strip(),
                                      a=ex["choices"][0], b=ex["choices"][1],
                                      c=ex["choices"][2], d=ex["choices"][3]),
            "coop_choice": "ABCD"[ex["answer"]],   # correct letter -> correctness label
            "seed": rng2.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "max_new_tokens": 512,
            "game_type": "mmlu",
            "framing": "knowledge_qa",
        })
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mmlu", type=int, default=150)
    ap.add_argument("--n-gsm8k", type=int, default=150)
    ap.add_argument("--out", default="data/runs/reason_steer/mixed_train.jsonl")
    args = ap.parse_args()

    gsm = [json.loads(l) for l in
           Path("data/runs/reason_steer/gsm8k_train.jsonl").read_text().splitlines() if l.strip()]
    gsm = gsm[:args.n_gsm8k]
    mmlu = build_mmlu_offset(args.n_mmlu)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in gsm + mmlu:
            f.write(json.dumps(r) + "\n")
    print(f"mixed corpus: {len(gsm)} gsm8k + {len(mmlu)} mmlu = {len(gsm)+len(mmlu)} -> {out}")


if __name__ == "__main__":
    main()
