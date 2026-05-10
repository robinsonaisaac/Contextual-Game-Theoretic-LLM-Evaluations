"""Build MMLU, GPQA Diamond, and HLE-MCQA corpora for the steering pipeline.

All three are 4-option MCQA (A/B/C/D). The pipeline parses
`<decision>X</decision>` and treats `decision == coop_choice` as
"correct"; setting `coop_choice` to the ground-truth letter therefore
gives us per-α correctness rate directly.

Usage: python3 scripts/build_reasoning_corpora.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 11
N_MMLU = 300
N_GPQA = 198      # GPQA Diamond is 198 items
N_HLE = 250

PROMPT_4 = """{question}

A. {a}
B. {b}
C. {c}
D. {d}

Reason briefly, then end with <decision>A</decision>, <decision>B</decision>, <decision>C</decision>, or <decision>D</decision>."""


def build_mmlu(rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = list(range(len(ds)))
    rng.shuffle(items)
    items = items[:N_MMLU]
    out = []
    for idx in items:
        ex = ds[idx]
        choices = ex["choices"]
        if len(choices) != 4:
            continue
        coop = "ABCD"[ex["answer"]]
        prompt = PROMPT_4.format(
            question=ex["question"].strip(),
            a=choices[0], b=choices[1], c=choices[2], d=choices[3],
        )
        out.append({
            "story_id": f"mmlu_{idx:05d}",
            "prompt": prompt,
            "coop_choice": coop,
            "seed": rng.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "game_type": "mmlu",
            "framing": ex.get("subject", ""),
        })
    return out


def build_gpqa(rng: random.Random) -> list[dict]:
    """Use fingertap/GPQA-Diamond ungated mirror.

    The mirror's items have a single `question` field with multiple-choice
    options embedded inline (a/b/c/d) and a single `answer` letter A-D.
    We re-anchor the elicitation prompt for our `<decision>X</decision>`
    parser. The letter mapping is preserved.
    """
    from datasets import load_dataset
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    out = []
    for i, ex in enumerate(ds):
        question = (ex.get("question") or "").rstrip()
        answer = (ex.get("answer") or "").strip().upper()
        if not question or answer not in {"A", "B", "C", "D"}:
            continue
        prompt = (
            question
            + "\n\nReason briefly, then end with <decision>A</decision>, "
              "<decision>B</decision>, <decision>C</decision>, or "
              "<decision>D</decision>."
        )
        out.append({
            "story_id": f"gpqa_{i:04d}",
            "prompt": prompt,
            "coop_choice": answer,
            "seed": rng.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "game_type": "gpqa_diamond",
            "framing": "graduate_qa",
        })
    return out[:N_GPQA]


def build_hle(rng: random.Random) -> list[dict]:
    """HLE has mixed format. Filter to multi-choice items where the question
    contains 'Answer Choices:' with 4 options.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("cais/hle", split="test")
    except Exception as e:
        print(f"[hle] failed to load: {e}")
        return []
    out = []
    items = list(range(len(ds)))
    rng.shuffle(items)
    for idx in items:
        ex = ds[idx]
        # HLE uses answer_type field; "multipleChoice" indicates MCQA
        if ex.get("answer_type") != "multipleChoice":
            continue
        # The question itself contains the answer choices inline. The answer
        # is a single letter A-Z. We only handle 4-option items (A-D).
        question = ex.get("question", "")
        answer = (ex.get("answer") or "").strip().upper()
        if not answer or answer not in {"A", "B", "C", "D"}:
            continue
        # Use the question as-is (it embeds the choices) and map coop_choice.
        # We also re-anchor the elicitation prompt.
        prompt = (question.rstrip()
                  + "\n\nReason briefly, then end with <decision>A</decision>, "
                    "<decision>B</decision>, <decision>C</decision>, or "
                    "<decision>D</decision>.")
        out.append({
            "story_id": f"hle_{idx:05d}",
            "prompt": prompt,
            "coop_choice": answer,
            "seed": rng.randint(0, 2**16 - 1),
            "temperature": 0.7,
            "game_type": "hle_mcqa",
            "framing": ex.get("category", ""),
        })
        if len(out) >= N_HLE:
            break
    return out


def main():
    rng = random.Random(SEED)
    for name, builder in [("mmlu", build_mmlu),
                          ("gpqa_diamond", build_gpqa),
                          ("hle_mcqa", build_hle)]:
        print(f"\n[{name}] building...")
        try:
            rows = builder(rng)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        out = Path(f"data/runs/reasoning/{name}_eval.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        print(f"  {len(rows)} items -> {out}")
        print(f"  coop dist: {dict(Counter(r['coop_choice'] for r in rows))}")


if __name__ == "__main__":
    main()
