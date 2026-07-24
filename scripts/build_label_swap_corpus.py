"""Build a label-swapped version of the steering JSONL corpus.

For every story, swap A <-> B in:
- "Decision A" / "Decision B"
- "<decision>A</decision>" / "<decision>B</decision>"
- "labeled as A" / "labeled as B"

And invert `coop_choice` (A -> B, B -> A) so the swapped story still
labels the same SEMANTIC cooperative action correctly.

This is the diagnostic test: if a steering vector is encoding "cooperation",
its effect should be UNCHANGED on the swapped corpus (because we swap both
the surface label and the coop_choice). If it's encoding "prefer label A",
its measured cooperation effect should INVERT on the swapped corpus
(because what was A is now B).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Three-step swap to avoid the second replacement undoing the first.
SWAP_PAIRS = [
    ("Decision A", "Decision B"),
    ("<decision>A</decision>", "<decision>B</decision>"),
    ("labeled as A", "labeled as B"),
]


def swap_labels(text: str) -> str:
    out = text
    for a, b in SWAP_PAIRS:
        sentinel = f"\x00{hash(a) & 0xFFFF:04x}\x00"
        out = out.replace(a, sentinel)
        out = out.replace(b, a)
        out = out.replace(sentinel, b)
    return out


def swap_choice(c: str) -> str:
    return {"A": "B", "B": "A"}.get(c, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out-path", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.in_path) as fin, open(args.out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["story_id"] = obj["story_id"] + "_swap"
            obj["prompt"] = swap_labels(obj["prompt"])
            obj["coop_choice"] = swap_choice(obj["coop_choice"])
            fout.write(json.dumps(obj) + "\n")
            n += 1
    print(f"wrote {n} swapped stories to {args.out_path}")


if __name__ == "__main__":
    main()
