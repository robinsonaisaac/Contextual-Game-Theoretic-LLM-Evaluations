# scripts/fetch_bbh_tier5.py
"""Fetch 3 long-horizon BBH subsets into data/runs/bbh/ in the eval schema used by
scripts/tinker_eval.py --eval ledger (mechanical ANSWER: tail; gold in row['answer']).

RUN WITH:  .venv-tinker/bin/python scripts/fetch_bbh_tier5.py
(datasets is available in .venv-tinker 4.8.5). Answers are compared with normalized
exact-match; for dyck we store the closer sequence with whitespace removed and instruct
the model to output it without spaces.

NOTE: The brief specifies maveriq/bigbenchhard but its loading script is incompatible
with datasets>=4.0. We use lukaemon/bbh instead — identical data (same HF source),
compatible Parquet format. Both have 250 rows per subset."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

SUBSETS = ["multistep_arithmetic_two", "tracking_shuffled_objects_three_objects", "dyck_languages"]
TAIL = "\n\nThink step by step, then end with a line: ANSWER: <final answer>."
OUT = Path("data/runs/bbh")


def _load(subset):
    # maveriq/bigbenchhard uses a legacy loading script incompatible with datasets>=4.0;
    # lukaemon/bbh is identical data in Parquet format and works without any workaround.
    ds = load_dataset("lukaemon/bbh", subset)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        ds = _load(subset)
        rows = []
        for i, ex in enumerate(ds):
            gold = str(ex["target"]).strip()
            prompt_body = ex["input"].strip()
            if subset == "dyck_languages":
                gold = gold.replace(" ", "")
                prompt_body += ("\n\nGive ONLY the sequence of closing brackets needed to "
                                "close all open brackets, with NO spaces.")
            rows.append({
                "story_id": f"bbh_{subset}_{i:04d}",
                "prompt": prompt_body + TAIL,
                "answer": gold,
                "seed": i,
                "game_type": f"bbh_{subset}",
                "framing": subset,
                "max_new_tokens": 2048,
            })
        out = OUT / f"{subset}_eval.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[fetch_bbh] {subset}: wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
