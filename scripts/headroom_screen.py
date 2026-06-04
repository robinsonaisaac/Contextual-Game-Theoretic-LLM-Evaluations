"""Headroom screen: run base Qwen3-4B on each candidate eval set; KEEP those at
0.25 <= accuracy <= 0.80 (drop ceiling/floor — BBH-3 was ceiling, GPQA@512 floor).
Run: .venv-tinker/bin/python scripts/headroom_screen.py  (source .env first)
"""
import json, subprocess, sys
from pathlib import Path

BASE = "Qwen/Qwen3-4B-Instruct-2507"
SETS = {
    "heldout_family": ("freetext",  "data/runs/gt_rlvr/eval_heldout_family.jsonl"),
    "depth_extrap":   ("freetext",  "data/runs/gt_rlvr/eval_depth_extrap.jsonl"),
    "dyck":           ("dyck",      "data/runs/gt_rlvr/eval_dyck.jsonl"),
    "prontoqa":       ("prontoqa",  "data/runs/gt_rlvr/eval_prontoqa.jsonl"),
    "mmlu_pro":       ("mmlu_pro",  "data/runs/gt_rlvr/eval_mmlu_pro.jsonl"),
    "bbh_hard":       ("bbh_hard",  "data/runs/gt_rlvr/eval_bbh_hard.jsonl"),
    "gsm8k":          ("gsm8k",     "data/runs/capability/gsm8k_eval.jsonl"),
}


def main():
    keep, report = [], {}
    for name, (kind, corpus) in SETS.items():
        if not Path(corpus).exists():
            print(f"{name:16s} MISSING {corpus}"); continue
        out = f"data/runs/gt_rlvr/screen_{name}.json"
        subprocess.run([sys.executable, "scripts/tinker_eval.py", "--eval", kind,
                        "--corpus", corpus, "--base-model", BASE, "--tokenizer", BASE,
                        "--limit", "80", "--out", out], check=True)
        acc = json.load(open(out))["accuracy"]
        verdict = "KEEP" if 0.25 <= acc <= 0.80 else "DROP (ceiling/floor)"
        report[name] = acc
        print(f"{name:16s} acc={acc:.2f}  {verdict}")
        if 0.25 <= acc <= 0.80:
            keep.append(name)
    Path("data/runs/gt_rlvr/screened.json").write_text(json.dumps({"keep": keep, "acc": report}, indent=2))
    print("\nKEEP:", keep)
    print("(gsm8k kept regardless as a no-regression check)")


if __name__ == "__main__":
    main()
