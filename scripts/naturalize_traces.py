# scripts/naturalize_traces.py
"""Sonnet naturalization of canonical ledger traces + extract-back round-trip QC.

RUN WITH SYSTEM python3, after loading the MAIN repo .env (never print the key):
  set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
  python3 scripts/naturalize_traces.py --smoke

Rephraser + extractor model: anthropic/claude-sonnet-4-6 (OpenRouter). Every naturalized
trace must round-trip: extract-back the (key,value) facts + answer and mechanically check
SET equality vs the gold skeleton (order NOT enforced). Reject-and-regenerate (max 2
retries, rotating the style seed) then drop. Corrupted traces would teach corrupted
bookkeeping, so QC is non-negotiable."""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

MODEL = "anthropic/claude-sonnet-4-6"
STYLE_SEEDS = [
    "an engineer's log",
    "a formal derivation",
    "casual working notes",
    "a markdown state table",
    "numbered facts F1/F2/F3...",
]

REPHRASE_SYSTEM = (
    "You rewrite structured reasoning logs into natural, varied working notes. You MUST "
    "preserve every fact exactly: never change any key, value, or dependency; never add or "
    "drop a fact; keep the final answer identical. Only the surface wording and structure "
    "may change. Do NOT solve anything yourself — only restyle what is given."
)

REPHRASE_USER = (
    "Rewrite the following solution log in the style of: {style}.\n"
    "Preserve every noted fact (key = value) and the final answer EXACTLY. You may reorder "
    "prose and reword freely, but every key and its value must still be clearly stated, and "
    "each fact's dependencies (what it was computed from) must remain recoverable. Keep a "
    "final line that states the answer.\n\n"
    "---\n{canonical}\n---\n\n"
    "Return ONLY the rewritten log."
)

EXTRACT_SYSTEM = (
    "You extract structured facts from a reasoning log. Respond ONLY with a JSON object, no "
    "prose, exactly: {\"notes\": [[key, value], ...], \"answer\": \"...\"}. List every fact "
    "the log establishes as [key, value]. If a fact was corrected during the log, report the "
    "FINAL corrected value. 'answer' is the final answer stated in the log."
)

EXTRACT_USER = "Extract the facts and final answer from this log as strict JSON.\n\n---\n{text}\n---"


def qc_ok(ex_notes: set, ex_answer: str, gold_facts: dict, gold_answer: str) -> bool:
    gold = {(str(k).strip(), str(v).strip()) for k, v in gold_facts.items()}
    return ex_notes == gold and str(ex_answer).strip() == str(gold_answer).strip()


def _client():
    import openai
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set — source the main-repo .env first")
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _rephrase(client, canonical: str, style: str) -> str:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7, max_tokens=2048,
        messages=[{"role": "system", "content": REPHRASE_SYSTEM},
                  {"role": "user", "content": REPHRASE_USER.format(style=style, canonical=canonical)}])
    return r.choices[0].message.content or ""


def _extract_back(client, text: str) -> Tuple[set, str]:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=2048,
        messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                  {"role": "user", "content": EXTRACT_USER.format(text=text)}])
    raw = (r.choices[0].message.content or "{}").strip()
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    raw = raw.rstrip("`").strip()
    try:
        d = json.loads(raw)
        notes = {(str(k).strip(), str(v).strip()) for k, v in d.get("notes", [])}
        return notes, str(d.get("answer", "")).strip()
    except Exception:
        return set(), ""


def _naturalize_one(client, item: dict) -> dict:
    for attempt in range(3):        # 1 try + 2 retries
        style = STYLE_SEEDS[(item["style_idx"] + attempt) % len(STYLE_SEEDS)]
        try:
            text = _rephrase(client, item["canonical"], style)
            exn, exa = _extract_back(client, text)
        except Exception:
            continue
        if qc_ok(exn, exa, item["gold_facts"], item["gold_answer"]):
            return {"prompt": item["prompt"], "completion": text, "ok": True,
                    "family": item["family"], "style": style}
    return {"prompt": item["prompt"], "completion": None, "ok": False, "family": item["family"]}


def naturalize_all(items: List[dict], workers: int = 8) -> Tuple[List[dict], float]:
    client = _client()
    out: List[Optional[dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_naturalize_one, client, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    results = [r for r in out if r is not None]
    passed = sum(1 for r in results if r["ok"])
    return results, (passed / len(items) if items else 0.0)


def _smoke():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from game_theory_llm.reasoning.ledger_protocol import render_canonical
    from game_theory_llm.reasoning.ledger_tasks import get_generator
    rng = random.Random(0)
    items = []
    for i, fam in enumerate(["trees", "register_machine", "graph_search", "forward_chain",
                             "trees", "register_machine", "graph_search", "forward_chain",
                             "trees", "graph_search"]):
        t = get_generator(fam)(seed=1000 + i, horizon=rng.choice([12, 18, 24]))
        items.append({"prompt": t.prompt, "canonical": render_canonical(t),
                      "gold_facts": t.gold_facts, "gold_answer": t.gold_answer,
                      "family": t.family, "style_idx": i})
    results, rate = naturalize_all(items, workers=8)
    print(f"[naturalize] smoke n={len(items)} QC pass-rate={rate:.3f}")
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] family={r['family']}")
    # Print one full passing example
    for r in results:
        if r["ok"] and r.get("completion"):
            print("\n--- EXAMPLE NATURALIZED TRACE ---")
            print(f"family: {r['family']}  style: {r.get('style', 'unknown')}")
            print(r["completion"][:1500])
            print("--- END EXAMPLE ---")
            break
    assert rate >= 0.70, f"QC pass-rate {rate:.3f} < 0.70 — investigate rephraser fidelity"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
