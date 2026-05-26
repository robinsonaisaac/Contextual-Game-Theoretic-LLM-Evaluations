"""Convert generated trust vignettes into the steering JSONL format.

Reads the per-story .txt files written by ``generate_trust_stories.py``
and emits a single train.jsonl + eval.jsonl + full_corpus.jsonl in the
exact format the existing steering pipeline expects (story_id, prompt,
coop_choice, seed, temperature, framing). Additional metadata fields
(cell_id, breakeven_p) ride along for downstream calibration-curve
analysis.

By convention coop_choice = "A" = Trust on every story (matches the
generator template).

Usage:
    python3 scripts/build_trust_steering_corpus.py \\
        --src-dir data/runs/trust_v1/stories \\
        --out-train data/runs/trust_v1/steering/train.jsonl \\
        --out-eval  data/runs/trust_v1/steering/eval.jsonl \\
        --out-full  data/runs/trust_v1/steering/full_corpus.jsonl \\
        --eval-frac 0.16 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from game_theory_llm.trust_games import TRUST_GAMES


def _cells_by_id() -> dict:
    return {c.cell_id: c for c in TRUST_GAMES}


def _parse_filename(stem: str) -> tuple[str, str, int]:
    """Parse `trust__{cell_id}__{framing}__{k:03d}` -> (cell_id, framing, k)."""
    parts = stem.split("__")
    if len(parts) != 4 or parts[0] != "trust":
        raise ValueError(f"unexpected story filename: {stem}")
    return parts[1], parts[2], int(parts[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--out-full", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = _cells_by_id()
    rows = []
    for path in sorted(Path(args.src_dir).glob("trust__*.txt")):
        cell_id, framing, k = _parse_filename(path.stem)
        cell = cells.get(cell_id)
        if cell is None:
            print(f"  skipping {path.name}: unknown cell {cell_id}")
            continue
        text = path.read_text().strip()
        if not text:
            print(f"  skipping {path.name}: empty file")
            continue
        rows.append({
            "story_id": f"trust__{cell_id}__{framing}__{k:03d}",
            "prompt": text,
            "coop_choice": "A",          # A = Trust, by template construction
            "seed": hash((cell_id, framing, k)) & 0xFFFF,
            "temperature": 0.7,
            "framing": framing,
            "cell_id": cell_id,
            "breakeven_p": round(cell.breakeven_p, 4),
            "R": cell.R, "P": cell.P, "S": cell.S,
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = max(1, int(round(len(rows) * args.eval_frac)))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    for path_s, subset in [
        (args.out_full,  rows),
        (args.out_train, train_rows),
        (args.out_eval,  eval_rows),
    ]:
        out = Path(path_s)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")

    print(json.dumps({
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_cells": len({r["cell_id"] for r in rows}),
        "out_full": args.out_full,
    }, indent=2))


if __name__ == "__main__":
    main()
