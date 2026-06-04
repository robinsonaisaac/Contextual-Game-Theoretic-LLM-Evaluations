"""Emit GRPO train prompts + held-out eval JSONLs for the RLVR transfer probe.

Train families: level_k + iterated_dominance (depths 2-4).
HELD-OUT transfer family: bargaining (high-entropy 0-100 answers -> clean transfer signal).
Eval also adds depth-extrapolation (depths 5-6 of trained families) and the non-game
depth-scaled diagnostics (Dyck, ProntoQA).
"""
from __future__ import annotations

import json
from pathlib import Path

from game_theory_llm.reasoning.freetext import bargaining, level_k, iterated_dominance
from game_theory_llm.reasoning.eval_gen import dyck, prontoqa

OUT = Path("data/runs/gt_rlvr")
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, rows):
    for r in rows:
        r.setdefault("max_new_tokens", 256 + 256 * int(r.get("depth", 3)))
    (OUT / name).write_text("\n".join(json.dumps(r) for r in rows))
    return len(rows)


def main():
    # TRAIN: level_k + iterated_dominance, depths 2-4 (bargaining held out)
    train = [fam(seed=10_000 * d + i, depth=d)
             for fam in (level_k, iterated_dominance)
             for d in (2, 3, 4) for i in range(200)]
    n_train = dump("train.jsonl", train)

    counts = {"train": n_train}
    counts["heldout_family"] = dump("eval_heldout_family.jsonl",
        [bargaining(seed=90_000 * d + i, depth=d) for d in (2, 3, 4) for i in range(40)])
    counts["depth_extrap"] = dump("eval_depth_extrap.jsonl",
        [fam(seed=70_000 * d + i, depth=d)
         for fam in (level_k, iterated_dominance) for d in (5, 6) for i in range(25)])
    counts["dyck"] = dump("eval_dyck.jsonl",
        [dyck(seed=i, depth=d) for d in (2, 3, 4, 5, 6) for i in range(30)])
    counts["prontoqa"] = dump("eval_prontoqa.jsonl",
        [prontoqa(seed=i, depth=d) for d in (2, 3, 4, 5) for i in range(30)])
    print(counts)


if __name__ == "__main__":
    main()
