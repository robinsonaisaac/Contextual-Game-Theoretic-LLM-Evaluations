"""Steps 1+2 via OpenRouter (sampling-only; Tinker key revoked server-side).

Same logic as memory_harness.py / decomp_harness.py but over OpenRouter chat completions
(model qwen/qwen3-30b-a3b-instruct-2507, temperature 0). To keep ±harness comparisons
within-provider, run the --mode plain controls here too (do NOT mix with Tinker numbers).

Modes:
  plain  : single-call control (8192 tokens)
  memory : stateless-with-memory multi-round harness (notes survive, text does not)
  decomp : env-driven post-order traversal; model answers only local min/max queries

RUN (py3.9, OPENROUTER_API_KEY in env):
  python3 scripts/harness_or.py --mode memory --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl \
      --out data/runs/gt_rlvr/t4or_harness_base_d6.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
NOTE = re.compile(r"^\s*NOTE:\s*(\S{1,40})\s*=\s*(.{1,40}?)\s*$", re.MULTILINE)
ANS_INT = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
ANS_DYCK = re.compile(r"<answer>\s*([)\]}>]+)\s*</answer>")
SID = re.compile(r"gt_d(\d+)_(\d+)$")
SEED_BASE = {6: 76000, 7: 77000}
MAX_NOTES = 300

INSTR = """
=== MULTI-ROUND MODE WITH EXTERNAL MEMORY ===
You are solving this in up to {R} rounds. IMPORTANT: your reasoning text is NOT carried
between rounds — only your saved notes survive. Anything you do not save is lost.

To save a fact, output a line of exactly this form (one per line, integer or short value):
NOTE: <key> = <value>
For example, to record that the subtree reached by moves A then B is worth -3:
NOTE: A.B = -3

Work on a manageable chunk this round, save what you conclude as notes, then end your
reply with the single word CONTINUE. In a later round, combine saved notes to finish.
When you know the final answer, output it as <answer>...</answer> (this ends the task).

This is round {r} of {R}. Your saved notes so far:
{table}
"""

# GUIDED: the explicit bottom-up caching strategy spelled out (the RLVR target). Brackets the
# step-3 ceiling — is the strategy *operable* by this model once known?
GUIDED = """
=== EXTERNAL MEMORY — FOLLOW THIS EXACT ALGORITHM ===
Your reasoning text is NOT carried between rounds. ONLY notes survive. Save with lines:
NOTE: <path> = <value>     (e.g.  NOTE: A.B = -3)

Backward induction as a NOTES-CACHING loop (do NOT try to solve the whole tree in your head):
1. A node's path is its move sequence from the root (root = empty, its children A and B, etc.).
   The deepest nodes are the leaf outcomes printed in the problem.
2. THIS round: find every node ALL of whose children are already values you can read directly
   (leaves from the problem, or already in your notes table). For each such node, the player to
   move is MAX at even path-length, MIN at odd path-length. Save NOTE: <path> = max/min(children).
   Do a few of these per round. Do NOT recompute notes you already have.
3. Once you have NOTE for both A and B, the answer is max(A, B). Output <answer>VALUE</answer>.

Be careful with whose turn it is (MAX at even depth from root, MIN at odd). Use ONLY values
you can read; never estimate. This is round {r} of {R}. Your saved notes so far:
{table}
"""

import sys
_spec = importlib.util.spec_from_file_location(
    "gametree", Path(__file__).resolve().parent.parent / "game_theory_llm/reasoning/gametree.py")
gametree = importlib.util.module_from_spec(_spec)
sys.modules["gametree"] = gametree     # required for dataclass annotation resolution
_spec.loader.exec_module(gametree)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def chat(prompt: str, max_tokens: int, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def score(text, row):
    if isinstance(row["answer"], str):
        m = ANS_DYCK.search(text or "")
        return (m is not None and m.group(1) == row["answer"], m is not None)
    m = ANS_INT.search(text or "")
    return (m is not None and int(m.group(1)) == int(row["answer"]), m is not None)


def run_plain(row, max_tokens=8192, **_):
    text = chat(row["prompt"], max_tokens)
    c, p = score(text, row)
    return {"story_id": row["story_id"], "depth": row.get("depth"), "correct": c, "parsed": p}


def run_memory(row, rounds=10, turn_tokens=2000, template=INSTR, **_):
    notes, last_text, r = {}, "", 0
    for r in range(1, rounds + 1):
        table = "\n".join(f"  {k} = {v}" for k, v in notes.items()) or "  (none yet)"
        last_text = chat(row["prompt"] + template.format(R=rounds, r=r, table=table), turn_tokens)
        for k, v in NOTE.findall(last_text):
            if len(notes) < MAX_NOTES or k in notes:
                notes[k] = v
        if ANS_INT.search(last_text) or ANS_DYCK.search(last_text):
            break
    c, p = score(last_text, row)
    return {"story_id": row["story_id"], "depth": row.get("depth"), "correct": c, "parsed": p,
            "rounds_used": r, "n_notes": len(notes),
            "notes": dict(list(notes.items())[:80]), "final_text_tail": last_text[-600:]}


def run_decomp(row, **_):
    m = SID.match(row["story_id"])
    depth, idx = int(m.group(1)), int(m.group(2))
    root = gametree._gen_tree(random.Random(SEED_BASE[depth] + idx), depth, 2, -9, 9)
    assert gametree.minimax(root, True) == row["answer"], row["story_id"]
    stats = {"nodes": 0, "local_correct": 0, "parse_fail": 0}

    def ask(maximizing, child_vals):
        who = "MAX" if maximizing else "MIN"
        goal = "the LARGEST outcome" if maximizing else "the SMALLEST outcome"
        opts = ", ".join(f"{lab} leads to outcome {v:+d}" for lab, v in child_vals)
        out = chat(f"A one-step choice in a two-player game. It is {who}'s turn; {who} wants "
                   f"{goal}. The available moves: {opts}. Under {who}'s optimal choice, what "
                   f"is the resulting outcome value? Answer only <answer>NUMBER</answer>.", 256)
        mm = ANS_INT.search(out)
        truth = (max if maximizing else min)(v for _, v in child_vals)
        stats["nodes"] += 1
        if mm is None:
            stats["parse_fail"] += 1
            return 0
        val = int(mm.group(1))
        if val == truth:
            stats["local_correct"] += 1
        return val

    def solve(node, maximizing):
        if node.is_leaf:
            return node.value
        return ask(maximizing, [(lab, solve(c, not maximizing)) for lab, c in node.children])

    pred = solve(root, True)
    return {"story_id": row["story_id"], "depth": depth, "correct": pred == row["answer"],
            "local_acc": stats["local_correct"] / max(1, stats["nodes"]),
            "nodes": stats["nodes"], "parse_fail": stats["parse_fail"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["plain", "memory", "guided", "decomp"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--turn-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    fn = {"plain": run_plain, "memory": run_memory, "guided": run_memory,
          "decomp": run_decomp}[args.mode]
    template = GUIDED if args.mode == "guided" else INSTR
    print(f"[{args.mode}@openrouter] {len(rows)} items on {MODEL} (temp 0)")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(lambda row: fn(row, rounds=args.rounds,
                                         turn_tokens=args.turn_tokens,
                                         max_tokens=args.max_tokens,
                                         template=template), rows))
    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    extra = ""
    if args.mode == "decomp":
        la = sum(r["local_acc"] * r["nodes"] for r in res) / sum(r["nodes"] for r in res)
        extra = f" local_acc={la:.4f} slip_pred={la ** res[0]['nodes']:.3f}"
    if args.mode == "memory":
        extra = (f" mean_rounds={sum(r['rounds_used'] for r in res)/n:.1f}"
                 f" mean_notes={sum(r['n_notes'] for r in res)/n:.1f}")
    print(f"[{args.mode}@openrouter] n={n} accuracy={acc:.3f}{extra}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"mode": f"{args.mode}_openrouter", "model": MODEL, "n": n, "accuracy": acc,
         "per_item": res}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
