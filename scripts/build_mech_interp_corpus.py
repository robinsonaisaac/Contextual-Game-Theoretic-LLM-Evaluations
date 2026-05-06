"""Build activation-extraction corpus for mechanistic interpretability experiments.

Extends build_steering_corpus.py to cover all 7 games × 5 contrast dims × 2 levels,
embedding game_type / contrast_dim / contrast_dim_level into each story record so
that offline probing can recover these labels from the saved bundles.

Cell filenames follow the pattern: {game}__{contrast_dim}__{level}.jsonl
e.g. prisoners_dilemma__gender__female.jsonl

Usage:
    python3 scripts/build_mech_interp_corpus.py \\
        --src-dir data/runs/2026-05-05-sharp/stories \\
        --out     data/runs/mech_interp/stories.jsonl \\
        --n-per-cell 50 --seed 0

Then run extraction:
    python3 scripts/run_steering.py extract \\
        --run-id mech-interp-v1 --stories data/runs/mech_interp/stories.jsonl --split train
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

GAMES = {
    "prisoners_dilemma", "stag_hunt", "chicken", "pure_coordination",
    "harmony", "battle_of_the_sexes", "matching_pennies", "deadlock",
}
CONTRAST_DIMS = {"gender", "realism", "era", "contrast_domain", "observability"}


def _parse_cell_id(stem: str) -> tuple[str, str, str] | None:
    """Parse '{game}__{contrast_dim}__{level}' from a JSONL filename stem.

    Returns (game_type, contrast_dim, level) or None if the stem doesn't match.
    """
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    game, dim, level = parts
    if game not in GAMES or dim not in CONTRAST_DIMS:
        return None
    return game, dim, level


def _load_all_cells(src_dir: Path, n_per_cell: int | None, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    n_cells = 0
    n_skipped = 0
    for path in sorted(src_dir.glob("*.jsonl")):
        parsed = _parse_cell_id(path.stem)
        if parsed is None:
            n_skipped += 1
            continue
        game_type, contrast_dim, level = parsed
        n_cells += 1

        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if n_per_cell is not None:
            # Sample deterministically so reruns are stable.
            indices = list(range(len(lines)))
            rng.shuffle(indices)
            indices = indices[:n_per_cell]
            indices.sort()
            lines = [lines[i] for i in indices]

        for i, line in enumerate(lines):
            obj = json.loads(line)
            rows.append({
                "story_id": f"{path.stem}_{i:03d}",
                "prompt": obj["content"],
                "coop_choice": "A",          # focal decision is always A in sharp design
                "seed": hash((path.stem, i)) & 0xFFFF,
                "temperature": 0.7,
                # Mech-interp metadata — stored in bundle.metadata during extraction
                "game_type": game_type,
                "contrast_dim": contrast_dim,
                "contrast_dim_level": level,
                "cell_id": path.stem,
            })

    print(f"[corpus] {n_cells} cells loaded, {n_skipped} non-matching files skipped")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True,
                    help="Directory containing {game}__{dim}__{level}.jsonl files")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path (all stories in one file)")
    ap.add_argument("--n-per-cell", type=int, default=None,
                    help="Max stories per cell. Default: all stories.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = _load_all_cells(Path(args.src_dir), args.n_per_cell, rng)
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    game_counts = Counter(r["game_type"] for r in rows)
    dim_counts = Counter(r["contrast_dim"] for r in rows)
    print(json.dumps({
        "n_total": len(rows),
        "out": str(out),
        "by_game": dict(sorted(game_counts.items())),
        "by_dim": dict(sorted(dim_counts.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
