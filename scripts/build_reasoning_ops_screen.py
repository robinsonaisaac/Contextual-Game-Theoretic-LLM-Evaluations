"""Build per-depth screen corpora for the new reasoning-operation generators.

Confirms each new op has measurable headroom on the 30B base (and finds the per-op
depth band for a breadth curriculum). One jsonl per op, depths 1..6, 24/depth.
Run: python3 scripts/build_reasoning_ops_screen.py
"""
import json
from pathlib import Path

from game_theory_llm.reasoning.reasoning_ops import nim_grundy, opponent_id, signal_abduce

OUT = Path("data/runs/gt_rlvr")
GENS = {"nim_grundy": nim_grundy, "opponent_id": opponent_id, "signal_abduce": signal_abduce}
DEPTHS = range(1, 7)
PER = 24


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, fn in GENS.items():
        rows = []
        for d in DEPTHS:
            for i in range(PER):
                r = fn(seed=500 * d + i, depth=d)
                r["max_new_tokens"] = 1024 + 256 * d
                rows.append(r)
        (OUT / f"screen_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        counts[name] = len(rows)
    print(counts)


if __name__ == "__main__":
    main()
