"""Subsample rationality train + heldout corpora for sprint-mode runs.

Train: 100 stories per game (PD, Harmony, Deadlock) -> 300 total.
Heldout: 50 per game (Stag Hunt, Chicken, BoS) -> 150 total.

Usage:
    python3 scripts/subsample_rationality.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/runs/2026-05-05-sharp/rationality")
SEED = 0
TRAIN_PER_GAME = 100
HELDOUT_PER_GAME = 50


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _balance(rows: list[dict], per_game: int, rng: random.Random) -> list[dict]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_game[r["game_type"]].append(r)
    out = []
    for game, items in by_game.items():
        rng.shuffle(items)
        out.extend(items[:per_game])
    rng.shuffle(out)
    return out


def main():
    rng = random.Random(SEED)
    train = _read(ROOT / "train.jsonl")
    heldout = _read(ROOT / "heldout_eval.jsonl")

    train_subset = _balance(train, TRAIN_PER_GAME, rng)
    heldout_subset = _balance(heldout, HELDOUT_PER_GAME, rng)

    (ROOT / "train_subset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_subset) + "\n"
    )
    (ROOT / "heldout_eval_subset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in heldout_subset) + "\n"
    )

    from collections import Counter
    print(json.dumps({
        "train_subset": {
            "n": len(train_subset),
            "by_game": dict(sorted(Counter(r["game_type"] for r in train_subset).items())),
        },
        "heldout_subset": {
            "n": len(heldout_subset),
            "by_game": dict(sorted(Counter(r["game_type"] for r in heldout_subset).items())),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
