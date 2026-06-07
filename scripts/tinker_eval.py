"""Evaluate a Tinker model (base or fine-tuned checkpoint) on the reasoning suite.

Samples each prompt via the Tinker sampling client (concurrently), decodes, and
scores locally:
  * gametree : strict <answer>N</answer> vs gold; reports accuracy + truncation
               per depth and per band (in-band vs extrapolation).
  * gsm8k    : numeric match (capability_scoring.gsm8k_correct).
  * mmlu/bbh : <decision>A-D</decision> == gold letter (accuracy-among-parsed too).

RUN WITH:  .venv-tinker/bin/python scripts/tinker_eval.py --eval gametree \
             --corpus data/runs/gametree/sft_eval.jsonl \
             [--model-path tinker://...  | --base-model Qwen/Qwen3-4B-Instruct-2507] \
             --tokenizer Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from tinker import ModelInput
from tinker.types import SamplingParams
from transformers import AutoTokenizer

_DEC = re.compile(r"<decision>\s*([A-D])\s*</decision>")
_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_float(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def gsm8k_correct(text, gold, tol=1e-4):
    """<answer> tag, else last number in text; compare to gold (self-contained,
    mirrors game_theory_llm.capability_scoring to avoid importing the heavy pkg)."""
    m = _ANS.search(text or "")
    pred = _to_float(m.group(1)) if m else (_to_float(_NUM.findall(text)[-1]) if _NUM.findall(text or "") else None)
    g = gold if isinstance(gold, (int, float)) else _to_float(gold)
    return pred is not None and g is not None and abs(pred - g) <= tol * max(1.0, abs(g))


import re as _re
_ANS_INT = _re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
_ANS_TF = _re.compile(r"<answer>\s*(True|False)\s*</answer>", _re.IGNORECASE)
_ANS_DYCK = _re.compile(r"<answer>\s*([)\]}>]+)\s*</answer>")
_DEC_J = _re.compile(r"<decision>\s*([A-J])\s*</decision>")


def score(eval_kind, text, row):
    """Return (correct: bool, parsed: bool)."""
    if eval_kind == "gametree":
        m = _ANS.search(text)
        return (m is not None and int(m.group(1)) == row["gametree_gold"], m is not None)
    if eval_kind == "gsm8k":
        ok = gsm8k_correct(text, row["gsm8k_gold"])
        return (ok, _ANS.search(text) is not None)
    if eval_kind == "freetext":                 # game-theory free-text: integer in row["answer"]
        m = _ANS_INT.search(text)
        return (m is not None and int(m.group(1)) == int(row["answer"]), m is not None)
    if eval_kind == "dyck":                      # exact closer string in row["answer"]
        m = _ANS_DYCK.search(text)
        return (m is not None and m.group(1) == row["answer"], m is not None)
    if eval_kind == "prontoqa":                  # True/False in row["answer"]
        m = _ANS_TF.search(text)
        return (m is not None and m.group(1).capitalize() == row["answer"], m is not None)
    if eval_kind == "mmlu_pro":                  # 10-way letter A-J in row["coop_choice"]
        m = _DEC_J.search(text)
        return (m is not None and m.group(1) == row["coop_choice"], m is not None)
    if eval_kind == "bbh_hard":                  # normalized exact-match of final answer
        m = _ANS.search(text) or _DEC.search(text) or _re.search(r"<answer>\s*(.+?)\s*</answer>", text)
        pred = m.group(1).strip() if m else None
        gold = str(row["answer"]).strip()
        norm = lambda s: s.lower().strip().strip(".()").strip()
        return (pred is not None and norm(pred) == norm(gold), m is not None)
    # mmlu / bbh (legacy A-D)
    m = _DEC.search(text)
    return (m is not None and m.group(1) == row["coop_choice"], m is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    choices=["gametree", "gsm8k", "mmlu", "bbh",
                             "freetext", "dyck", "prontoqa", "mmlu_pro", "bbh_hard"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-path", default=None, help="tinker:// fine-tuned checkpoint")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 => use per-row max_new_tokens or 1024")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sc = tinker.ServiceClient()
    smp = (sc.create_sampling_client(model_path=args.model_path) if args.model_path
           else sc.create_sampling_client(base_model=args.base_model))
    tag = args.model_path or args.base_model
    print(f"[eval] {args.eval}: {len(rows)} prompts on {tag}")

    def run_one(row):
        text = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
        mx = args.max_tokens or row.get("max_new_tokens", 1024)
        fut = smp.sample(prompt=ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=SamplingParams(max_tokens=mx, temperature=args.temperature))
        out = tok.decode(fut.result().sequences[0].tokens, skip_special_tokens=True)
        c, p = score(args.eval, out, row)
        return {"story_id": row.get("story_id"), "depth": row.get("depth"),
                "band": row.get("band"), "family": row.get("family"),
                "op_tags": row.get("op_tags"), "knowledge": row.get("knowledge"),
                "correct": c, "parsed": p}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(run_one, rows))

    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    pr = sum(r["parsed"] for r in res) / n
    print(f"\n[eval] {args.eval}: n={n} accuracy={acc:.3f} parse_rate={pr:.3f}")
    if args.eval in ("mmlu", "bbh", "gsm8k"):
        acp = sum(r["correct"] for r in res if r["parsed"]) / max(1, sum(r["parsed"] for r in res))
        print(f"       accuracy-among-parsed={acp:.3f}")
    if args.eval == "gametree":
        by = defaultdict(list)
        for r in res:
            by[r["depth"]].append(r)
        print(f"  {'depth':>5} {'band':>14} {'n':>3} {'acc':>6} {'trunc':>6}")
        for d in sorted(by):
            g = by[d]
            acc_d = sum(x["correct"] for x in g) / len(g)
            trunc = 1 - sum(x["parsed"] for x in g) / len(g)
            print(f"  {d:>5} {g[0]['band']:>14} {len(g):>3} {acc_d:>6.3f} {trunc:>6.3f}")
    out = args.out or f"data/runs/gametree/eval_{args.eval}_{'ft' if args.model_path else 'base'}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"eval": args.eval, "model": tag, "n": n,
                                     "accuracy": acc, "parse_rate": pr, "per_item": res}, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
