"""Build steering JSONL corpus from generated PD story files.

For each story dict in the source JSONL files we emit one line:
    {"story_id": ..., "prompt": ..., "coop_choice": "A",
     "seed": ..., "temperature": 0.7,
     "framing": <source-stem>}

`coop_choice` is always "A" because the PD generator pins
`focal_decision="A"` (see game_theory_llm/games.py).

Usage:
    python3 scripts/build_steering_corpus.py \\
        --src-dir data/runs/2026-05-05-sharp/stories \\
        --game prisoners_dilemma \\
        --out-train data/runs/2026-05-05-sharp/steering/train.jsonl \\
        --out-eval  data/runs/2026-05-05-sharp/steering/eval.jsonl \\
        --eval-frac 0.16 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_stories(src_dir: Path, game: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(src_dir.glob(f"{game}__*.jsonl")):
        framing = path.stem.removeprefix(f"{game}__")
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({
                    "story_id": f"{path.stem}_{i:03d}",
                    "prompt": obj["content"],
                    "coop_choice": "A",
                    "seed": hash((path.stem, i)) & 0xFFFF,
                    "temperature": 0.7,
                    "framing": framing,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--game", default="prisoners_dilemma")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, take only this many total stories (for quick smoke runs).")
    args = ap.parse_args()

    rows = _load_stories(Path(args.src_dir), args.game)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    n_eval = max(1, int(round(len(rows) * args.eval_frac)))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    for path_s, subset in [(args.out_train, train_rows), (args.out_eval, eval_rows)]:
        out = Path(path_s)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")

    print(json.dumps({
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "out_train": args.out_train,
        "out_eval": args.out_eval,
    }, indent=2))


if __name__ == "__main__":
    main()
