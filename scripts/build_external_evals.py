"""Load MMLU-Pro + BBH-Hard reasoning subtasks into eval JSONLs for the probe.

MMLU-Pro: 10-way MCQA (A-J), prompt ends with <decision>X</decision>; coop_choice=gold letter.
BBH-Hard: selected hard subtasks; prompt ends with <answer>ANSWER</answer>; row["answer"]=gold
          (normalized exact-match in tinker_eval bbh_hard scorer).
Run: python3 scripts/build_external_evals.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

OUT = Path("data/runs/gt_rlvr"); OUT.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCDEFGHIJ"
BBH_SUBTASKS = ["multistep_arithmetic_two", "web_of_lies",
                "tracking_shuffled_objects_seven_objects",
                "logical_deduction_seven_objects", "geometric_shapes"]


def build_mmlu_pro(n=300):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = list(range(len(ds))); random.Random(0).shuffle(idx)
    rows = []
    for i in idx[: n * 2]:
        ex = ds[i]
        opts = ex["options"]
        if not (2 <= len(opts) <= 10):
            continue
        gold = ex["answer"] if ex["answer"] in LETTERS else LETTERS[ex["answer_index"]]
        body = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        prompt = (f"{ex['question']}\n{body}\n\nReason step by step, then end with "
                  f"<decision>LETTER</decision> (one of {', '.join(LETTERS[:len(opts)])}).")
        rows.append({"story_id": f"mmlupro_{i}", "prompt": prompt, "coop_choice": gold,
                     "max_new_tokens": 1024, "category": ex.get("category", "")})
        if len(rows) >= n:
            break
    (OUT / "eval_mmlu_pro.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return len(rows)


def build_bbh_hard(per=60):
    from datasets import load_dataset
    rows = []
    for sub in BBH_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", sub, split="test")
        except Exception as e:
            print(f"  bbh {sub}: load failed ({type(e).__name__}); skipping")
            continue
        idx = list(range(len(ds))); random.Random(0).shuffle(idx)
        for i in idx[:per]:
            ex = ds[i]
            gold = str(ex["target"]).strip()
            prompt = (f"{ex['input']}\n\nReason step by step, then give your final answer as "
                      f"<answer>ANSWER</answer>.")
            rows.append({"story_id": f"bbh_{sub}_{i}", "prompt": prompt, "answer": gold,
                         "subtask": sub, "max_new_tokens": 1024})
    (OUT / "eval_bbh_hard.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return len(rows)


if __name__ == "__main__":
    print("mmlu_pro:", build_mmlu_pro())
    print("bbh_hard:", build_bbh_hard())
