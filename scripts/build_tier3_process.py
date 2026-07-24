"""Tier-3: process-reward vs outcome-reward head-to-head (pre-registered).

Hypothesis (from Tier-1b/2): outcome-only GRPO teaches depth-specific shortcuts —
in-domain gains, zero extrapolation. Process rewards (exact intermediate node values
from the solver, free by construction) reward the *procedure*, which is what could
generalize.

Design — two arms IDENTICAL except the reward function:
  * arm OUTCOME : reward = 1{final answer correct}            (existing env)
  * arm PROCESS : reward = 0.5*outcome + 0.5*process_score    (order-aware recall of
                  the gold internal-node value sequence, spray-guarded)
Both train on the EXACT Tier-1b problem set (gametree d3:200 / d4:500 / d5:500,
curriculum-ordered) at 6144 training tokens. Matching Tier-1b's problems also lets the
new OUTCOME arm isolate Tier-1b's 2048-token confound.

Pre-registered endpoints:
  * PRIMARY   : depth-6 extrapolation accuracy @ 8192 eval tokens (n=160), process vs
                outcome vs base. Dissociates procedure-learning from shortcut-learning.
  * SECONDARY : in-domain d3-5 (n=40/depth), held-out transfer (boolean/dyck/mmlu/bbh,
                reusing t2 base numbers at matched budgets), gsm8k no-regression,
                training frac_all_bad (process should have fewer dead groups).

Run (py3.9): python3 scripts/build_tier3_process.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from game_theory_llm.reasoning import gametree
from game_theory_llm.reasoning.process_reward import gold_process_values

OUT = Path("data/runs/gt_rlvr")
SID = re.compile(r"gt_d(\d+)_(\d+)$")


def main():
    # ---- TRAIN: Tier-1b problems + process_gold (identical prompts, both arms) ----
    rows = [json.loads(l) for l in open(OUT / "train_tier1b.jsonl")]
    for r in rows:
        m = SID.match(r["story_id"])
        depth, seed = int(m.group(1)), int(m.group(2))
        gold = gold_process_values(seed=seed, depth=depth)
        assert gold[-1] == r["answer"], r["story_id"]      # root value == final answer
        r["process_gold"] = gold
    (OUT / "train_tier3.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    # ---- PRIMARY endpoint: d6 extrapolation, n=160 (extends the Tier-1b d6 seed range) ----
    d6 = []
    for i in range(160):
        p = gametree.gen_game_tree_problem(seed=70000 + 6000 + i, depth=6, branching=2,
                                           framing="negotiation")
        d6.append({"story_id": f"gt_d6_{i}", "prompt": p["prompt"], "answer": p["answer"],
                   "depth": 6, "family": "gametree", "op_tags": ["backward_induction"],
                   "max_new_tokens": 8192})
    (OUT / "eval_t3_d6.jsonl").write_text("\n".join(json.dumps(r) for r in d6))

    # ---- SECONDARY: in-domain (fresh seeds, trained depths) ----
    indom = []
    for d in (3, 4, 5):
        for i in range(40):
            p = gametree.gen_game_tree_problem(seed=95000 + 1000 * d + i, depth=d,
                                               branching=2, framing="negotiation")
            indom.append({"story_id": f"gt_d{d}_i{i}", "prompt": p["prompt"],
                          "answer": p["answer"], "depth": d, "family": "gametree",
                          "op_tags": ["backward_induction"], "max_new_tokens": 8192})
    (OUT / "eval_t3_indomain.jsonl").write_text("\n".join(json.dumps(r) for r in indom))

    print(f"train_tier3.jsonl      n={len(rows)} (Tier-1b problems + process_gold)")
    print(f"eval_t3_d6.jsonl       n={len(d6)} (PRIMARY: d6 extrapolation)")
    print(f"eval_t3_indomain.jsonl n={len(indom)}")


if __name__ == "__main__":
    main()
