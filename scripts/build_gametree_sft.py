"""Build the game-tree SFT train set + a depth-stratified eval set.

Train: depths in the TRAIN band, each example = (prompt, gold backward-induction
CoT completion). Eval: in-band depths + EXTRAPOLATION depths (deeper than train)
with verifiable answers — the extrapolation split is the real long-depth test.
Seeds are offset so train/eval are disjoint.

Per the design review: also emits a CONTROL train set (`--control`) of
structure-matched but reasoning-trivial depth-1 trees padded to the same CoT
length, to isolate the game-theory contribution from generic CoT practice.

Usage:
    python3 scripts/build_gametree_sft.py            # train + eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_theory_llm.reasoning.gametree import gen_game_tree_sft, gen_game_tree_problem

FRAMINGS = ["abstract", "negotiation", "chess"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-depths", default="2,3,4")
    ap.add_argument("--extrap-depths", default="5,6,7")
    ap.add_argument("--n-train", type=int, default=200)   # per depth
    ap.add_argument("--n-eval", type=int, default=50)     # per depth
    ap.add_argument("--out", default="data/runs/gametree")
    args = ap.parse_args()
    train_d = [int(x) for x in args.train_depths.split(",")]
    extrap_d = [int(x) for x in args.extrap_depths.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # TRAIN: gold-CoT SFT examples (seeds 0..n_train-1 per depth)
    train = []
    for d in train_d:
        for i in range(args.n_train):
            p = gen_game_tree_sft(seed=10_000 * d + i, depth=d, framing=FRAMINGS[i % 3])
            train.append({"prompt": p["prompt"], "completion": p["completion"],
                          "answer": p["answer"], "depth": d})
    (out / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in train))

    # EVAL: in-band + extrapolation, verifiable answers (seeds 90000+ -> disjoint)
    ev = []
    for d in train_d + extrap_d:
        for i in range(args.n_eval):
            p = gen_game_tree_problem(seed=90_000 * d + i + 5, depth=d, framing=FRAMINGS[i % 3])
            ev.append({"story_id": p["story_id"], "prompt": p["prompt"],
                       "coop_choice": "A", "gametree_gold": p["answer"],
                       "depth": d, "band": "train" if d in train_d else "extrapolation",
                       "max_new_tokens": 256 + 256 * d, "game_type": "gametree"})  # budget scales w/ depth
    (out / "sft_eval.jsonl").write_text("\n".join(json.dumps(r) for r in ev))

    print(f"train: {len(train)} examples over depths {train_d}")
    print(f"eval:  {len(ev)} problems over depths {train_d}(in-band) + {extrap_d}(extrapolation)")
    print(f"-> {out}/sft_train.jsonl, {out}/sft_eval.jsonl")


if __name__ == "__main__":
    main()
