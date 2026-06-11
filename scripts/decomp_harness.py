"""Step-2 ablation: enforced decomposition — the harness drives the traversal, the model
only does local min/max steps.

Localizes the deficit exactly: the environment performs post-order traversal of the true
game tree (rebuilt from the item's seed) and at each internal node asks the model ONE
local question (pick max or min of the children's values), propagating the MODEL'S OWN
previous answers upward. If local competence is intact, end-to-end accuracy should return
to the slip regime p_local^(2^d - 1) — proving the d6 cliff is state/control, not local
computation.

Reports: per-node local accuracy, end-to-end root accuracy, slip-model prediction.

RUN: .venv-tinker/bin/python scripts/decomp_harness.py --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl \
       --base-model ... --tokenizer ... --limit 80 --out ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from tinker import ModelInput
from tinker.types import SamplingParams
from transformers import AutoTokenizer

# load gametree.py directly (pure stdlib) without triggering the heavy package __init__
_spec = importlib.util.spec_from_file_location(
    "gametree", Path(__file__).resolve().parent.parent / "game_theory_llm/reasoning/gametree.py")
gametree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gametree)

ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
SID = re.compile(r"gt_d(\d+)_(\d+)$")
# seed scheme used by build_tier3_process.py / build_step_evals: story_id gt_d{depth}_{i}
SEED_BASE = {6: 76000, 7: 77000}


def local_prompt(maximizing, child_vals):
    who = "MAX" if maximizing else "MIN"
    goal = "the LARGEST outcome" if maximizing else "the SMALLEST outcome"
    opts = ", ".join(f"{lab} leads to outcome {v:+d}" for lab, v in child_vals)
    return (f"A one-step choice in a two-player game. It is {who}'s turn; {who} wants "
            f"{goal}. The available moves: {opts}. Under {who}'s optimal choice, what is "
            f"the resulting outcome value? Answer only <answer>NUMBER</answer>.")


def run_item(row, tok, smp, max_tokens, temperature):
    m = SID.match(row["story_id"])
    depth, idx = int(m.group(1)), int(m.group(2))
    seed = SEED_BASE[depth] + idx
    root = gametree._gen_tree(random.Random(seed), depth, 2, -9, 9)
    assert gametree.minimax(root, True) == row["answer"], row["story_id"]

    stats = {"nodes": 0, "local_correct": 0, "parse_fail": 0}

    def ask(maximizing, child_vals):
        prompt = local_prompt(maximizing, child_vals)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
        fut = smp.sample(prompt=ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=SamplingParams(max_tokens=max_tokens,
                                                        temperature=temperature))
        out = tok.decode(fut.result().sequences[0].tokens, skip_special_tokens=True)
        mm = ANS.search(out)
        truth = (max if maximizing else min)(v for _, v in child_vals)
        stats["nodes"] += 1
        if mm is None:
            stats["parse_fail"] += 1
            return truth * 0      # degenerate fallback: 0 (counts as local error unless truth==0)
        val = int(mm.group(1))
        if val == truth:
            stats["local_correct"] += 1
        return val                # propagate the MODEL'S value, right or wrong

    def solve(node, maximizing):
        if node.is_leaf:
            return node.value
        child_vals = [(lab, solve(c, not maximizing)) for lab, c in node.children]
        return ask(maximizing, child_vals)

    pred = solve(root, True)
    return {"story_id": row["story_id"], "depth": depth,
            "correct": pred == row["answer"],
            "local_acc": stats["local_correct"] / max(1, stats["nodes"]),
            "nodes": stats["nodes"], "parse_fail": stats["parse_fail"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sc = tinker.ServiceClient()
    smp = (sc.create_sampling_client(model_path=args.model_path) if args.model_path
           else sc.create_sampling_client(base_model=args.base_model))
    print(f"[decomp] {len(rows)} items (each = 2^d-1 sequential local queries)")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(lambda row: run_item(row, tok, smp, args.max_tokens,
                                               args.temperature), rows))

    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    la = sum(r["local_acc"] * r["nodes"] for r in res) / sum(r["nodes"] for r in res)
    nodes = res[0]["nodes"]
    print(f"[decomp] n={n} end2end={acc:.3f} local_acc={la:.4f} "
          f"slip-model-predicted end2end={la ** nodes:.3f} (nodes={nodes})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"mode": "decomp_harness", "model": args.model_path or args.base_model, "n": n,
         "accuracy": acc, "local_acc": la, "slip_prediction": la ** nodes,
         "per_item": res}, indent=2))
    print(f"[decomp] wrote {args.out}")


if __name__ == "__main__":
    main()
