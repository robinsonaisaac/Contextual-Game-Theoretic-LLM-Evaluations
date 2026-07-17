"""Tier-5 attribution tooling: invariant-usage judge + suppression ablation.

Subcommands
-----------
judge       Read raw completions jsonl, judge each with Sonnet, report rates.
suppression Compare normal vs suppressed accuracy files, report shrinkage.

After both parts run, writes data/runs/tier5/attribution.json.

Usage (OpenRouter, system python3):
  python3 scripts/tier5_attribution.py judge \\
      --samples data/runs/tier5/raw_base_heldout_tracking_n30.jsonl \\
      --tag base_heldout_tracking

  python3 scripts/tier5_attribution.py suppression \\
      --normal  data/runs/tier5/t5_sft_heldout_tracking.json \\
      --suppressed data/runs/tier5/t5_sftsupp_heldout_tracking.json \\
      --tag heldout_tracking \\
      --base-acc 0.780 --sft-acc 0.080
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INVARIANTS = ["externalizes", "grounds_steps", "small_steps", "restates",
              "reads_off_answer", "uses_ledger"]

ATTRIBUTION_PATH = Path("data/runs/tier5/attribution.json")

JUDGE_SYSTEM = """\
You are an expert evaluator of LLM reasoning strategies.

Given a model response to a reasoning task, classify whether the response uses
an explicit *ledger* bookkeeping technique — where the model externalises its
working state into a compact written record that it updates and reads from,
rather than reasoning purely in implicit chain-of-thought.

Respond ONLY with a single JSON object (no markdown fences, no prose) with
exactly these six boolean fields:

{
  "externalizes": <bool>,
  "grounds_steps": <bool>,
  "small_steps": <bool>,
  "restates": <bool>,
  "reads_off_answer": <bool>,
  "uses_ledger": <bool>
}

Field definitions:
- externalizes: The response writes facts/state down compactly as found,
  rather than holding them implicitly in the reasoning stream.
- grounds_steps: Each computation step explicitly states which recorded facts
  it reads from, rather than operating on remembered values.
- small_steps: The response proceeds in small bounded incremental steps rather
  than attempting to re-derive long chains at once.
- restates: The response periodically re-states the full known-state as a
  checkpoint/summary before continuing.
- reads_off_answer: The final answer is read directly off the recorded state,
  not recomputed from scratch at the end.
- uses_ledger: True iff the response *overall* clearly uses the explicit
  ledger technique (not just incidental note-taking). This is the summary
  field. Set to true only if at least 3 of the above are true.

Be strict: brief in-passing mentions of numbers do NOT count as ledger use.
"""


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------
def _parse_judge(raw: str) -> dict:
    """Parse JSON judge response, default all to False on failure."""
    out = {k: False for k in INVARIANTS}
    try:
        s = (raw or "").strip()
        for fence in ("```json", "```"):
            if s.startswith(fence):
                s = s[len(fence):]
        s = s.rstrip("`").strip()
        d = json.loads(s)
        for k in INVARIANTS:
            out[k] = bool(d.get(k, False))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Judge one completion
# ---------------------------------------------------------------------------
def judge_one(client, text: str) -> dict:
    """Call Sonnet to classify a model output against the ledger invariants."""
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=128,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Model response to classify:\n\n{text}"},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return _parse_judge(raw)


# ---------------------------------------------------------------------------
# Attribution JSON helpers
# ---------------------------------------------------------------------------
def _load_attribution() -> dict:
    if ATTRIBUTION_PATH.exists():
        try:
            return json.loads(ATTRIBUTION_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_attribution(d: dict) -> None:
    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATTRIBUTION_PATH.write_text(json.dumps(d, indent=2))


# ---------------------------------------------------------------------------
# cmd_judge
# ---------------------------------------------------------------------------
def cmd_judge(args):
    """Run judge subcommand: read samples jsonl, judge all, report rates."""
    from openai import OpenAI  # available in system python3 via pip

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[judge] ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    samples_path = Path(args.samples)
    rows = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    n = len(rows)
    print(f"[judge] {args.tag}: judging {n} samples from {samples_path}")

    results = []
    failed = 0

    def _judge_row(row):
        try:
            return judge_one(client, row.get("text", ""))
        except Exception as e:
            print(f"[judge] WARNING: judge_one failed: {e}", file=sys.stderr)
            return {k: False for k in INVARIANTS}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_judge_row, rows))

    # Compute rates
    rates = {}
    for inv in INVARIANTS:
        rates[inv] = sum(r[inv] for r in results) / n if n else 0.0

    rates["n"] = n

    print(f"\n[judge] === {args.tag} ===")
    for inv in INVARIANTS:
        print(f"  {inv:20s}: {rates[inv]:.3f}  ({sum(r[inv] for r in results)}/{n})")

    # Persist
    attr = _load_attribution()
    attr.setdefault("part_a_ledger_usage_rates", {})[args.tag] = rates
    _save_attribution(attr)
    print(f"[judge] wrote {ATTRIBUTION_PATH}")


# ---------------------------------------------------------------------------
# cmd_suppression
# ---------------------------------------------------------------------------
def cmd_suppression(args):
    """Run suppression subcommand: compare normal vs suppressed accuracy files."""
    normal_path = Path(args.normal)
    supp_path = Path(args.suppressed)

    normal_data = json.loads(normal_path.read_text())
    supp_data = json.loads(supp_path.read_text())

    sft_suppressed = supp_data["accuracy"]
    sft_acc = args.sft_acc
    base_acc = args.base_acc

    shrinkage = sft_acc - sft_suppressed  # positive = suppression helped harm
    recovery = sft_suppressed - sft_acc   # positive = accuracy recovered toward base

    # Verdict logic
    gap_to_base = abs(sft_suppressed - base_acc)
    gap_to_sft = abs(sft_suppressed - sft_acc)

    if gap_to_base <= 0.15:
        verdict = "format_imposition — knowledge intact, ledger habit destructive"
    elif gap_to_sft <= 0.15:
        verdict = "catastrophic_forgetting — fine-tune damaged underlying capability"
    else:
        verdict = "partial — mixed format imposition and capability damage"

    print(f"\n[suppression] === {args.tag} ===")
    print(f"  base_acc     : {base_acc:.3f}")
    print(f"  sft_acc      : {sft_acc:.3f}  (harm: {sft_acc - base_acc:+.3f})")
    print(f"  sft_suppressed: {sft_suppressed:.3f}  (recovery: {sft_suppressed - sft_acc:+.3f})")
    print(f"  verdict      : {verdict}")

    entry = {
        "base": base_acc,
        "sft": sft_acc,
        "sft_suppressed": sft_suppressed,
        "shrinkage_by_suppression": round(sft_suppressed - sft_acc, 4),
        "verdict": verdict,
    }

    attr = _load_attribution()
    attr.setdefault("part_b_suppression", {})[args.tag] = entry
    _save_attribution(attr)
    print(f"[suppression] wrote {ATTRIBUTION_PATH}")


# ---------------------------------------------------------------------------
# cmd_finalize — write causal_verdict and print summary
# ---------------------------------------------------------------------------
def cmd_finalize(args):
    """Compute causal verdict from Part B results and finalize attribution.json."""
    attr = _load_attribution()
    part_b = attr.get("part_b_suppression", {})
    if not part_b:
        print("[finalize] ERROR: no part_b_suppression data found", file=sys.stderr)
        sys.exit(1)

    verdicts = [v["verdict"] for v in part_b.values()]

    if all("format_imposition" in v for v in verdicts):
        causal = "format_imposition — knowledge intact, ledger habit destructive"
    elif all("catastrophic_forgetting" in v for v in verdicts):
        causal = "catastrophic_forgetting — fine-tune damaged underlying capability"
    else:
        causal = "partial — mixed format imposition and capability damage"

    attr["causal_verdict"] = causal
    _save_attribution(attr)

    print(f"\n[finalize] causal_verdict: {causal}")
    print(f"[finalize] wrote {ATTRIBUTION_PATH}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tier-5 attribution tooling")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # judge subcommand
    p_judge = sub.add_parser("judge", help="Judge raw completions for ledger invariants")
    p_judge.add_argument("--samples", required=True, help="jsonl with story_id/text/family")
    p_judge.add_argument("--tag", required=True, help="key for attribution.json")

    # suppression subcommand
    p_supp = sub.add_parser("suppression", help="Compare normal vs suppressed accuracy")
    p_supp.add_argument("--normal", required=True, help="json from standard SFT eval")
    p_supp.add_argument("--suppressed", required=True, help="json from suppressed SFT eval")
    p_supp.add_argument("--tag", required=True, help="key for attribution.json")
    p_supp.add_argument("--base-acc", type=float, required=True)
    p_supp.add_argument("--sft-acc", type=float, required=True)

    # finalize subcommand
    p_fin = sub.add_parser("finalize", help="Compute causal verdict and finalize attribution.json")

    args = ap.parse_args()

    if args.cmd == "judge":
        cmd_judge(args)
    elif args.cmd == "suppression":
        cmd_suppression(args)
    elif args.cmd == "finalize":
        cmd_finalize(args)


if __name__ == "__main__":
    main()
