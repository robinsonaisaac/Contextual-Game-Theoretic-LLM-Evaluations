"""Generate rich trust-game vignettes via the project's LLMClient.

Mirrors the methodology of the existing PD corpus: an LLM writes narrative
scenarios that *embed* a trust dilemma rather than just present the payoff
matrix as a table. Each generated story is a fictional first-person
narrative ending in a binary choice (Decision A = Trust, Decision B =
Don't Trust). The payoff structure is woven into the narrative explicitly
in the final paragraph so the model can do EV reasoning if it wants.

Cells come from ``game_theory_llm.trust_games.TRUST_GAMES``; framings reuse
the five binary contrast dimensions from the PD generator so the trust
corpus is directly comparable.

Usage:
    python3 scripts/generate_trust_stories.py \\
        --out-dir data/runs/trust_v1/stories \\
        --model opus \\
        --n-per-cell 30 --seed 0

Cost: at $10/M tokens for Claude Opus and ~$1.5k tokens per story,
generating 10 cells × 30 stories = 300 stories costs roughly $5-10.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Iterable

from game_theory_llm.client import LLMClient
from game_theory_llm.trust_games import TRUST_GAMES, TrustGameConfig


FRAMINGS = [
    # (framing_id, descriptor injected into the prompt)
    ("business",   "set in a contemporary business or startup context"),
    ("personal",   "set in a personal-relationships context (friends, family, romantic partners, neighbours)"),
    ("political",  "set in a contemporary political or institutional context (coalition partners, regulators, journalists)"),
    ("military",   "set in a military, intelligence, or wartime alliance context"),
    ("historical", "set in a historically-grounded setting from any era before 1950"),
]


GENERATION_TEMPLATE = textwrap.dedent("""\
You are a fiction writer specialising in psychologically grounded vignettes about
trust. Write ONE self-contained scenario that ends in a binary trust dilemma.

Constraints (all required):
1. The story is {framing_descriptor}.
2. The protagonist must choose between trusting another character (Decision A) or
   not trusting them (Decision B). The other character has private information or
   power and may either reciprocate (honour the trust) or betray it.
3. The narrative must make the stakes feel real and plausible. Use 350-600 words.
4. The protagonist's name and the trustee's name must be distinct and culturally
   coherent with the framing.
5. The final 1-2 paragraphs must describe the payoff structure in concrete,
   in-world terms (not as a payoff table). Translate the abstract values
   (Reward = ${R}, Safe = ${P}, Sucker = ${S}) into stakes the characters care
   about (dollars, lives, careers, reputation, etc.) at roughly the right ratios.
   It is acceptable to use any units as long as the relative magnitudes match.
6. Do NOT mention game theory, payoff matrices, expected value, or probability
   explicitly.
7. End with this exact closing block, replacing PROTAGONIST_NAME:

You are PROTAGONIST_NAME in this scenario. What decision will you make?
Provide your justification for your decision inside <justification></justification> tags.
Then, output your decision, either: <decision>B</decision> or <decision>A</decision>.
Be sure to pay attention to which action is labeled as A and which is labeled as B, as they might not be in alphabetical order.

Output ONLY the vignette as plain text, no preamble, no JSON, no markdown
headings.
""")


async def generate_one(client: LLMClient, *, model_key: str,
                       cell: TrustGameConfig, framing: tuple[str, str],
                       seed: int) -> str:
    prompt = GENERATION_TEMPLATE.format(
        framing_descriptor=framing[1], R=cell.R, P=cell.P, S=cell.S,
    )
    # Seed-style variation: ask for the {seed}-th original idea so repeated
    # calls produce distinct stories.
    salted = prompt + f"\n\n(Variation seed: {seed}; produce a fresh scenario unlike any common-knowledge example.)"
    # LLMClient.generate returns {model_name: text|None}; unwrap.
    responses = await client.generate(salted, model=model_key)
    text = responses.get(model_key)
    if text is None:
        raise RuntimeError(f"LLMClient returned None for model={model_key}")
    return text


def _extract_text(raw: str) -> str:
    """Strip stray ```text or ```markdown fences if the model added them."""
    s = raw.strip()
    s = re.sub(r"^```[a-z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()


async def main_async(args) -> None:
    client = LLMClient()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    framings = FRAMINGS if not args.framings else [
        f for f in FRAMINGS if f[0] in args.framings.split(",")
    ]

    written = 0
    tasks: list[tuple[str, asyncio.Task]] = []

    sem = asyncio.Semaphore(args.concurrency)

    async def _job(out_path: Path, cell, framing, seed):
        async with sem:
            text = await generate_one(client, model_key=args.model,
                                      cell=cell, framing=framing, seed=seed)
            out_path.write_text(_extract_text(text))
            return out_path.name

    for cell in TRUST_GAMES:
        for framing_id, descriptor in framings:
            for k in range(args.n_per_cell):
                seed = rng.randint(0, 2**30)
                fname = f"trust__{cell.cell_id}__{framing_id}__{k:03d}.txt"
                out_path = out_dir / fname
                if out_path.exists() and not args.overwrite:
                    written += 1
                    continue
                tasks.append((fname, asyncio.create_task(
                    _job(out_path, cell, (framing_id, descriptor), seed),
                )))

    print(f"[trust-gen] dispatching {len(tasks)} stories (concurrency={args.concurrency})...",
          flush=True)
    for name, t in tasks:
        try:
            await t
            written += 1
            if written % 10 == 0:
                print(f"[trust-gen] wrote {written} so far...", flush=True)
        except Exception as e:
            print(f"[trust-gen] ERR on {name}: {type(e).__name__}: {e}", flush=True)

    print(f"[trust-gen] done; wrote {written} stories to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Directory to write per-story .txt files")
    ap.add_argument("--model", default="opus",
                    help="Key in client.MODEL_REGISTRY for the story-writing model (default opus)")
    ap.add_argument("--n-per-cell", type=int, default=6,
                    help="Stories per (payoff cell × framing). 6 × 10 cells × 5 framings = 300 total.")
    ap.add_argument("--framings", default="",
                    help="Comma-separated framing IDs to include; default = all five.")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, regenerate stories whose .txt files already exist.")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
