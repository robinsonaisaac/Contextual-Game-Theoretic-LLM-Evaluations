"""Build a BBH logical-deduction corpus formatted for the steering pipeline.

The existing pipeline parses `<decision>A</decision>` or `<decision>B</decision>`
only, so we convert each BBH 3-option question into a 2-option question by
keeping the correct answer plus one randomly chosen distractor, and randomly
assigning which is labeled (A) vs (B). `coop_choice` is set to the letter of
the correct answer.

Each emitted record matches the schema used by the steering pipeline:
    {"story_id", "prompt", "coop_choice", "seed", "temperature",
     "game_type": "bbh_logical_deduction", "framing": "<task>"}

Usage:
    python3 scripts/build_bbh_corpus.py \\
        --task logical_deduction_three_objects \\
        --out  data/runs/bbh/logical_deduction_eval.jsonl \\
        --n 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")
ANSWER_RE = re.compile(r"^\(([A-Z])\)\s*$")

PROMPT_TEMPLATE = """{stem}
Options:
(A) {opt_a}
(B) {opt_b}

Reason briefly, then give your answer as <decision>A</decision> or <decision>B</decision>."""


def _split_question(input_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split BBH 'input' into (stem_text, [(letter, option_text), ...])."""
    if "\nOptions:\n" not in input_text:
        raise ValueError(f"Unexpected BBH input format: {input_text[:200]}")
    stem, options_block = input_text.split("\nOptions:\n", 1)
    options = []
    for line in options_block.strip().splitlines():
        m = OPTION_RE.match(line.strip())
        if m:
            options.append((m.group(1), m.group(2).strip()))
    return stem.strip(), options


def _correct_letter(target: str) -> str:
    m = ANSWER_RE.match(target.strip())
    if not m:
        raise ValueError(f"Bad target: {target!r}")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="logical_deduction_three_objects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", args.task, split="test")
    rng = random.Random(args.seed)

    items = list(range(len(ds)))
    rng.shuffle(items)
    items = items[: args.n]

    rows: list[dict] = []
    for idx in items:
        ex = ds[idx]
        stem, options = _split_question(ex["input"])
        if len(options) < 2:
            continue
        opt_map = dict(options)             # letter -> text
        correct = _correct_letter(ex["target"])
        if correct not in opt_map:
            continue
        # Pair correct with one random distractor
        distractors = [l for l in opt_map if l != correct]
        if not distractors:
            continue
        distractor = rng.choice(distractors)

        # Random A/B assignment
        if rng.random() < 0.5:
            opt_a_text, opt_b_text = opt_map[correct], opt_map[distractor]
            coop_choice = "A"
        else:
            opt_a_text, opt_b_text = opt_map[distractor], opt_map[correct]
            coop_choice = "B"

        prompt = PROMPT_TEMPLATE.format(stem=stem, opt_a=opt_a_text, opt_b=opt_b_text)
        rows.append({
            "story_id": f"bbh_{args.task}_{idx:03d}",
            "prompt": prompt,
            "coop_choice": coop_choice,
            "seed": idx,
            "temperature": 0.7,
            "game_type": "bbh_logical_deduction",
            "framing": args.task,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(json.dumps({
        "task": args.task,
        "n_emitted": len(rows),
        "out": str(out),
        "coop_choice_distribution": dict(Counter(r["coop_choice"] for r in rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
