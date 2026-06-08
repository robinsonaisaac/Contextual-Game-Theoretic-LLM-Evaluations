"""Tier-2 breadth curriculum: deductive (game-tree) + inductive (opponent-ID).

Tier-1b showed depth-in-ONE-operation stays local. Tier-2 tests whether a *diverse*
curriculum transfers. The new-op screen found only search/bookkeeping operations retain
headroom on the 30B (closed-form nim/abduction are at ceiling), so the breadth here is the
two viable ones — backward_induction (gametree) and inductive_rule (opponent_id) — each
trained at its measurable depth band, with a LARGE token budget (both are generation-heavy;
their apparent floors were truncation walls).

Curriculum = difficulty tiers mixing both ops:  tier1 {gt d3, opp d1} -> tier2 {gt d4, opp d2}
-> tier3 {gt d5, opp d3}. Held-out extrapolation = strictly deeper of each op (gt d6, opp d4).

Run (py3.9):  python3 scripts/build_tier2_breadth.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from game_theory_llm.reasoning import gametree
from game_theory_llm.reasoning.reasoning_ops import opponent_id

OUT = Path("data/runs/gt_rlvr")
PER = 200          # per (op, depth) in train -> 2 ops x 3 depths x 200 = 1200
EXTRAP_PER = 80
INDOM_PER = 30
TIERS = [(3, 1), (4, 2), (5, 3)]     # (gametree depth, opponent depth) per difficulty tier


def gt(seed, depth):
    p = gametree.gen_game_tree_problem(seed=seed, depth=depth, branching=2, framing="negotiation")
    return {"story_id": f"gt_d{depth}_{seed}", "prompt": p["prompt"], "answer": p["answer"],
            "depth": depth, "family": "gametree", "op_tags": ["backward_induction"],
            "max_new_tokens": 512 + 256 * depth}


def opp(seed, depth):
    r = opponent_id(seed=seed, depth=depth)
    r["max_new_tokens"] = 8192
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # ---- TRAIN: curriculum by tier (ascending difficulty), both ops mixed within a tier ----
    train = []
    for ti, (gd, od) in enumerate(TIERS):
        block = [gt(30000 + 1000 * gd + i, gd) for i in range(PER)] + \
                [opp(40000 + 1000 * od + i, od) for i in range(PER)]
        rng.shuffle(block)            # mix the two ops within the tier; tiers stay ordered
        train.extend(block)
    (OUT / "train_tier2_breadth.jsonl").write_text("\n".join(json.dumps(r) for r in train))

    # ---- held-out depth-extrapolation (strictly deeper than trained) ----
    extrap = [gt(80000 + i, 6) for i in range(EXTRAP_PER)] + \
             [opp(81000 + i, 4) for i in range(EXTRAP_PER)]
    random.Random(1).shuffle(extrap)
    (OUT / "eval_tier2_extrap.jsonl").write_text("\n".join(json.dumps(r) for r in extrap))

    # ---- in-domain sanity (trained depths, fresh seeds) ----
    indom = []
    for gd, od in TIERS:
        indom += [gt(90000 + 1000 * gd + i, gd) for i in range(INDOM_PER)]
        indom += [opp(91000 + 1000 * od + i, od) for i in range(INDOM_PER)]
    random.Random(2).shuffle(indom)
    (OUT / "eval_tier2_indomain.jsonl").write_text("\n".join(json.dumps(r) for r in indom))

    n_gt = sum(1 for r in train if r["family"] == "gametree")
    print(f"train_tier2_breadth.jsonl  n={len(train)} (gametree={n_gt}, opponent={len(train)-n_gt})")
    print(f"eval_tier2_extrap.jsonl    n={len(extrap)} (gt d6 + opp d4, held-out deeper)")
    print(f"eval_tier2_indomain.jsonl  n={len(indom)}")


if __name__ == "__main__":
    main()
