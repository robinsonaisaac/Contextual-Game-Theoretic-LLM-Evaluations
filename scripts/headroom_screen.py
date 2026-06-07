"""Headroom screen (Gate 1): run base Qwen3-8B on each Tier-1 candidate benchmark; KEEP those
at 0.25 <= accuracy <= 0.80 (drop ceiling/floor). Operation-isolated knowledge-free suite +
knowledge-heavy contrast + GSM8k no-regression.
Run: .venv-tinker/bin/python scripts/headroom_screen.py  (source .env first)
"""
import json, subprocess, sys
from pathlib import Path

BASE = "Qwen/Qwen3-8B"
D = "data/runs/gt_rlvr"
SETS = {  # name: (eval_kind, corpus)
    "dyck":           ("dyck",     f"{D}/eval_dyck.jsonl"),
    "prontoqa":       ("prontoqa", f"{D}/eval_prontoqa.jsonl"),
    "countdown":      ("freetext", f"{D}/eval_countdown.jsonl"),
    "ordering":       ("freetext", f"{D}/eval_ordering.jsonl"),
    "knights_knaves": ("freetext", f"{D}/eval_knights_knaves.jsonl"),
    "boolean_eval":   ("freetext", f"{D}/eval_boolean_eval.jsonl"),
    "bbh_hard":       ("bbh_hard", f"{D}/eval_bbh_hard.jsonl"),
    "mmlu_pro":       ("mmlu_pro", f"{D}/eval_mmlu_pro.jsonl"),   # knowledge-heavy CONTRAST
    "gsm8k":          ("gsm8k",    "data/runs/capability/gsm8k_eval.jsonl"),  # no-regression
}


def main():
    keep, report = [], {}
    for name, (kind, corpus) in SETS.items():
        if not Path(corpus).exists():
            print(f"{name:16s} MISSING {corpus}"); continue
        out = f"{D}/screen8b_{name}.json"
        subprocess.run([sys.executable, "scripts/tinker_eval.py", "--eval", kind,
                        "--corpus", corpus, "--base-model", BASE, "--tokenizer", BASE,
                        "--limit", "80", "--out", out], check=True)
        acc = json.load(open(out))["accuracy"]
        verdict = "KEEP" if 0.25 <= acc <= 0.80 else "DROP(ceiling/floor)"
        report[name] = acc
        print(f"{name:16s} acc={acc:.2f}  {verdict}")
        if 0.25 <= acc <= 0.80:
            keep.append(name)
    Path(f"{D}/screened8b.json").write_text(json.dumps({"keep": keep, "acc": report}, indent=2))
    print("\nKEEP:", keep)


if __name__ == "__main__":
    main()
