"""Build the Tier-1b deep-curriculum training + held-out-depth extrapolation eval.

Tier-1 trained on *saturating* free-text families (level_k decays to 0; bargaining
converges to a fixed point; iterated_dominance / subtraction_game are one-step
formulas) — so "deeper" did not mean "harder", and the 30B base sat at ceiling.

Tier-1b centers the curriculum on **game-tree minimax backward induction**, which
genuinely scales: a depth-d, branching-b tree has b^d leaves and a non-saturating
integer answer, so difficulty rises monotonically with depth. This is also the family
that produced the original +12.5 pp depth-extrapolation effect at 4B. op_tag =
backward_induction (in TRAINED_TAGS), so the operation-overlap analysis still applies.

Outputs (TRAIN_DEPTHS / EXTRAP_DEPTHS are set from the 30B headroom screen):
  * train_tier1b.jsonl          curriculum-ordered (ascending depth) gametree problems
  * eval_depth_extrap_deep.jsonl held-out *deeper* depths (never trained) — the transfer test
  * eval_gametree_indomain.jsonl held-out items at trained depths — sanity (learned the op at all?)

Run (py3.9, no torch needed):  python3 scripts/build_tier1b.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from game_theory_llm.reasoning import gametree

OUT = Path("data/runs/gt_rlvr")
BRANCHING = 2

# Set from the gametree headroom screen on the 30B base (branching 2): a sharp capability
# cliff, NOT a ceiling — d3=1.00, d4=0.55, d5=0.15, d6+=0.00. GRPO needs within-group reward
# variance, which lives at d4-d5; d3 is the curriculum on-ramp. The held-out transfer test is
# d6-d7, currently floored at 0.00 — can curriculum-RLVR lift them off the floor (the 4B +12.5pp
# depth-extrapolation analog)?  Per-depth counts emphasize the learnable frontier (d4,d5).
TRAIN_DEPTHS = [3, 4, 5]                 # curriculum: on-ramp -> ideal-signal -> stretch
TRAIN_COUNTS = {3: 200, 4: 500, 5: 500}  # 1200 prompts (matches Tier-1 budget)
EXTRAP_DEPTHS = [6, 7]                    # never trained, base=0.00 -> depth-extrapolation test
PER_EXTRAP_DEPTH = 80                     # 160 total (matches Tier-1 extrap n)
PER_INDOMAIN_DEPTH = 30


def _mk(seed: int, depth: int) -> dict:
    p = gametree.gen_game_tree_problem(seed=seed, depth=depth, branching=BRANCHING,
                                       framing="negotiation")
    return {"story_id": f"gt_d{depth}_{seed}", "prompt": p["prompt"], "answer": p["answer"],
            "depth": depth, "family": "gametree", "framing": "negotiation",
            "op_tags": ["backward_induction"],
            "max_new_tokens": 512 + 256 * depth}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- TRAIN: curriculum-ordered by ascending depth (the rl.train loop reads batches
    #      sequentially, so file order == curriculum order). Within a depth, seeds vary. ----
    out_train = []
    rng = random.Random(0)
    for d in TRAIN_DEPTHS:                      # ascending depth == curriculum order
        block = [_mk(seed=20000 + 1000 * d + i, depth=d) for i in range(TRAIN_COUNTS[d])]
        rng.shuffle(block)                     # decorrelate seeds within a depth block
        out_train.extend(block)
    (OUT / "train_tier1b.jsonl").write_text("\n".join(json.dumps(r) for r in out_train))

    # ---- EXTRAP eval: strictly deeper, never-trained depths (the transfer test) ----
    extrap = [_mk(seed=70000 + 1000 * d + i, depth=d)
              for d in EXTRAP_DEPTHS for i in range(PER_EXTRAP_DEPTH)]
    random.Random(1).shuffle(extrap)
    (OUT / "eval_depth_extrap_deep.jsonl").write_text("\n".join(json.dumps(r) for r in extrap))

    # ---- in-domain held-out (trained depths, fresh seeds): did it learn the op at all? ----
    indom = [_mk(seed=90000 + 1000 * d + i, depth=d)
             for d in TRAIN_DEPTHS for i in range(PER_INDOMAIN_DEPTH)]
    random.Random(2).shuffle(indom)
    (OUT / "eval_gametree_indomain.jsonl").write_text("\n".join(json.dumps(r) for r in indom))

    print(f"train_tier1b.jsonl          n={len(out_train)}  depths={TRAIN_DEPTHS} (curriculum)")
    print(f"eval_depth_extrap_deep.jsonl n={len(extrap)}  depths={EXTRAP_DEPTHS} (held-out transfer)")
    print(f"eval_gametree_indomain.jsonl n={len(indom)}  depths={TRAIN_DEPTHS} (in-domain sanity)")


if __name__ == "__main__":
    main()
