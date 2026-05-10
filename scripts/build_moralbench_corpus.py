"""Build a MoralBench eval corpus formatted for the steering pipeline.

The pipeline parses `<decision>A</decision>` / `<decision>B</decision>` from
the model's response. MoralBench question files already contain the
"A. Agree B. Disagree" choices inline, so we just wrap them in our
elicitation prompt.

Each emitted record:
    {"story_id": "<dataset>_<foundation>_<i>",
     "prompt": "<question text>\n\nReason briefly...<decision>A/B</decision>",
     "coop_choice": "A",   # we report Agree-rate (A-rate) at eval time
     "seed": ...,
     "temperature": 0.7,
     "game_type": "moralbench_<dataset>",
     "framing": "<foundation>"}

Usage:
    python3 scripts/build_moralbench_corpus.py \\
        --src-dir /tmp/MoralBench \\
        --out data/runs/moralbench/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROMPT_SUFFIX = (
    "\n\nReason briefly, then end with <decision>A</decision> or "
    "<decision>B</decision>."
)


# Map foundation token in filename to a canonical label
FOUNDATION = {
    "harm":      "Care/Harm",
    "fairness":  "Fairness",
    "ingroup":   "Loyalty",
    "authority": "Authority",
    "purity":    "Sanctity",
    "liberty":   "Liberty",
}

ITEM_RE = re.compile(r"^([a-z]+)_(\d+)\.txt$")


def _load_dataset(src_dir: Path, dataset: str, answer_json: dict) -> list[dict]:
    rows: list[dict] = []
    qdir = src_dir / "questions" / dataset
    for path in sorted(qdir.glob("*.txt")):
        m = ITEM_RE.match(path.name)
        if m is None:
            continue
        foundation_token, idx = m.group(1), int(m.group(2))
        if foundation_token not in FOUNDATION:
            continue
        # Stem matches the answers JSON key (e.g. "harm_1")
        stem = f"{foundation_token}_{idx}"
        answer = answer_json.get(stem)

        text = path.read_text()
        prompt = text.strip() + PROMPT_SUFFIX

        rows.append({
            "story_id": f"{dataset}__{foundation_token}_{idx}",
            "prompt": prompt,
            "coop_choice": "A",     # report Agree-rate
            "seed": (hash((dataset, stem)) & 0xFFFF),
            "temperature": 0.7,
            "game_type": f"moralbench_{dataset}",
            "framing": FOUNDATION[foundation_token],
            "human_score_A": (answer or {}).get("A"),
            "human_score_B": (answer or {}).get("B"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="/tmp/MoralBench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="6_concepts,MFQ_30")
    args = ap.parse_args()

    src = Path(args.src_dir)
    rows: list[dict] = []
    for ds in args.datasets.split(","):
        ds = ds.strip()
        ans_path = src / "answers" / f"{ds}.json"
        if not ans_path.exists():
            print(f"[skip] no answer JSON for {ds}")
            continue
        with open(ans_path) as f:
            answer_json = json.load(f)
        sub = _load_dataset(src, ds, answer_json)
        print(f"[{ds}] {len(sub)} items")
        rows.extend(sub)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(json.dumps({
        "out": str(out),
        "n_total": len(rows),
        "by_dataset": dict(Counter(r["game_type"] for r in rows)),
        "by_foundation": dict(Counter(r["framing"] for r in rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
