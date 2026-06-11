"""Step-1 probe: stateless-with-memory agent harness for deep reasoning tasks.

Mechanism under test (from docs/results/rlvr_transfer_postmortem.md): the d6+ capacity
cliff is addressing/state-tracking — soft attention can't reliably retrieve 63 colliding
values from a long trace. This harness externalizes state into a keyed memory file:

  * Each ROUND the model sees ONLY: the problem + its saved notes table + instructions.
    The reasoning text of previous rounds is NOT carried over — state survives only via
    notes. (That is what makes it a memory file, not a longer transcript.)
  * The model saves notes with lines `NOTE: <key> = <value>` and ends a round with either
    CONTINUE or the final <answer>...</answer>.

Greedy decoding (eval-stochasticity lesson). Scores freetext ints and dyck strings.

RUN: .venv-tinker/bin/python scripts/memory_harness.py --corpus ... \
       [--base-model ... | --model-path tinker://...] --tokenizer ... --out ...
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from tinker import ModelInput
from tinker.types import SamplingParams
from transformers import AutoTokenizer

NOTE = re.compile(r"^\s*NOTE:\s*(\S{1,40})\s*=\s*(.{1,40}?)\s*$", re.MULTILINE)
ANS_INT = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
ANS_DYCK = re.compile(r"<answer>\s*([)\]}>]+)\s*</answer>")
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


def score(text, row):
    if isinstance(row["answer"], str):
        m = ANS_DYCK.search(text or "")
        return (m is not None and m.group(1) == row["answer"], m is not None)
    m = ANS_INT.search(text or "")
    return (m is not None and int(m.group(1)) == int(row["answer"]), m is not None)


def run_episode(row, tok, smp, rounds, turn_tokens, temperature):
    notes: dict = {}
    last_text = ""
    for r in range(1, rounds + 1):
        table = "\n".join(f"  {k} = {v}" for k, v in notes.items()) or "  (none yet)"
        prompt = row["prompt"] + INSTR.format(R=rounds, r=r, table=table)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
        fut = smp.sample(prompt=ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=SamplingParams(max_tokens=turn_tokens,
                                                        temperature=temperature))
        last_text = tok.decode(fut.result().sequences[0].tokens, skip_special_tokens=True)
        for k, v in NOTE.findall(last_text):
            if len(notes) < MAX_NOTES or k in notes:
                notes[k] = v
        if ANS_INT.search(last_text) or ANS_DYCK.search(last_text):
            break
    c, p = score(last_text, row)
    return {"story_id": row.get("story_id"), "depth": row.get("depth"),
            "family": row.get("family"), "correct": c, "parsed": p,
            "rounds_used": r, "n_notes": len(notes),
            "notes": dict(list(notes.items())[:80]), "final_text_tail": last_text[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--turn-tokens", type=int, default=2000)
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
    print(f"[harness] {len(rows)} items, rounds<={args.rounds}, "
          f"turn_tokens={args.turn_tokens}, temp={args.temperature}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(lambda row: run_episode(row, tok, smp, args.rounds,
                                                  args.turn_tokens, args.temperature), rows))

    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    pr = sum(r["parsed"] for r in res) / n
    mr = sum(r["rounds_used"] for r in res) / n
    mn = sum(r["n_notes"] for r in res) / n
    print(f"[harness] n={n} accuracy={acc:.3f} parse_rate={pr:.3f} "
          f"mean_rounds={mr:.1f} mean_notes={mn:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"mode": "memory_harness", "model": args.model_path or args.base_model,
         "rounds": args.rounds, "turn_tokens": args.turn_tokens, "n": n,
         "accuracy": acc, "parse_rate": pr, "per_item": res}, indent=2))
    print(f"[harness] wrote {args.out}")


if __name__ == "__main__":
    main()
