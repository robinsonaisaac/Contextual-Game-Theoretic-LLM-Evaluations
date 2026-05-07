"""Deterministically subsample a steering JSONL (and its swap-label twin)
to a smaller held-out set. Used to keep the cross-model + swap diagnostic
within tight wall-clock budgets while still covering all 7 alphas on every
model.

The selection seeds Python's `random` so the same seed reproduces the
same N stories. The swap corpus must already be aligned to the regular
one by `_swap`-suffixed `story_id`s (see `build_label_swap_corpus.py`).

Usage:
    python3 scripts/build_subset_corpus.py \\
        --regular data/runs/2026-05-05-sharp/steering/full_corpus.jsonl \\
        --swap    data/runs/2026-05-05-sharp/steering/full_corpus_swap.jsonl \\
        --out-regular data/runs/2026-05-05-sharp/steering/full_corpus_subset50.jsonl \\
        --out-swap    data/runs/2026-05-05-sharp/steering/full_corpus_swap_subset50.jsonl \\
        --n 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular", required=True)
    ap.add_argument("--swap", required=True)
    ap.add_argument("--out-regular", required=True)
    ap.add_argument("--out-swap", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full = [json.loads(l) for l in Path(args.regular).read_text().splitlines() if l.strip()]
    swap = [json.loads(l) for l in Path(args.swap).read_text().splitlines() if l.strip()]
    swap_by_id = {s["story_id"]: s for s in swap}

    rng = random.Random(args.seed)
    rng.shuffle(full)
    subset = full[: args.n]
    subset_swap = [swap_by_id[s["story_id"] + "_swap"] for s in subset]

    Path(args.out_regular).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_regular).write_text(
        "\n".join(json.dumps(s) for s in subset) + "\n"
    )
    Path(args.out_swap).write_text(
        "\n".join(json.dumps(s) for s in subset_swap) + "\n"
    )
    print(json.dumps({
        "n": len(subset),
        "n_swap": len(subset_swap),
        "out_regular": args.out_regular,
        "out_swap": args.out_swap,
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
