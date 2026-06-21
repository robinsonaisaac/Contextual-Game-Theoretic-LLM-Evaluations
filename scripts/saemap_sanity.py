"""Step-0 HARD GATE: generate-and-judge decision readout (3c pipeline).

The base model (Qwen3.5-9B-Base on Modal A100-80GB) generates a free-text
decision paragraph for each PD scenario. Claude Sonnet 4.6 (via OpenRouter)
grades each paragraph as cooperate / defect / unclear.

GATE passes when:
  - unclear_rate < 0.25  (model produces gradeable decisions)
  - 0.1 < coop_rate < 0.9  (model is content-responsive, not degenerate)

Old logit-based metrics (pcoop / ab_mass) are kept as cheap corroborating
signal but are no longer the gating criterion.
"""
import json, statistics
from dotenv import load_dotenv

# Load OpenRouter key from the main-repo .env BEFORE importing judge
load_dotenv("/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env")

from game_theory_llm.saemap import remote, corpus, paths
from game_theory_llm.saemap import judge

# ── Smoke test: 5 scenarios, print continuations + verdicts ────────────────

def smoke(rows):
    print("=" * 60)
    print("SMOKE TEST (first 5 scenarios)")
    print("=" * 60)
    smoke_rows = rows[:5]
    smoke_prompts = [corpus.decision_elicitation(r["prompt"]) for r in smoke_rows]
    smoke_conts = remote.generate(smoke_prompts, max_new_tokens=2000, temperature=0.0,
                                  stop_string="</decision>")
    smoke_items = [
        {"scenario": r["prompt"], "continuation": c, "coop_letter": r["coop_letter"]}
        for r, c in zip(smoke_rows, smoke_conts)
    ]
    smoke_verdicts = judge.grade_decisions(smoke_items)
    n_gradeable = sum(1 for v in smoke_verdicts if v["verdict"] != "unclear")
    for i, (r, cont, verd) in enumerate(zip(smoke_rows, smoke_conts, smoke_verdicts)):
        trunc = cont[:300].replace("\n", " ").strip()
        print(f"\n[{i+1}] {r['id']} (coop={r['coop_letter']})")
        print(f"  cont: {trunc}{'...' if len(cont) > 300 else ''}")
        print(f"  verdict: {verd['verdict']}")
    print(f"\nSmoke gradeable: {n_gradeable}/5")
    if n_gradeable < 4:
        print("WARNING: fewer than 4/5 graded — elicitation may need tuning")
    print("=" * 60)
    return n_gradeable


def main():
    paths.ensure_run_dirs()

    rows = corpus.pd_eval_set(limit=40)

    # ── Smoke first ─────────────────────────────────────────────────────────
    n_gradeable_smoke = smoke(rows)

    # ── Full 40-scenario generate+judge gate ─────────────────────────────────
    print("\nRunning full gate (40 scenarios)...")
    prompts = [corpus.decision_elicitation(r["prompt"]) for r in rows]
    conts = remote.generate(prompts, max_new_tokens=2000, temperature=0.0,
                            stop_string="</decision>")

    items = [
        {"scenario": r["prompt"], "continuation": c, "coop_letter": r["coop_letter"]}
        for r, c in zip(rows, conts)
    ]
    verdicts = judge.grade_decisions(items)

    n_coop    = sum(1 for v in verdicts if v["verdict"] == "cooperate")
    n_defect  = sum(1 for v in verdicts if v["verdict"] == "defect")
    n_unclear = sum(1 for v in verdicts if v["verdict"] == "unclear")
    n         = len(rows)

    coop_rate    = n_coop / max(1, n_coop + n_defect)
    unclear_rate = n_unclear / n

    gate_pass = (unclear_rate < 0.25) and (0.1 < coop_rate < 0.9)

    # ── Old logit metrics for cross-reference (cheap) ────────────────────────
    coop_letters = [r["coop_letter"] for r in rows]
    try:
        ps_logit = remote.pcoop(prompts, coop_letters).tolist()
        mean_p_coop = statistics.mean(ps_logit)
        std_p_coop  = statistics.pstdev(ps_logit)
    except Exception as e:
        print(f"[warn] pcoop cross-reference failed: {e}")
        mean_p_coop = None
        std_p_coop  = None

    rec = {
        "n": n,
        "n_coop": n_coop,
        "n_defect": n_defect,
        "n_unclear": n_unclear,
        "coop_rate": coop_rate,
        "unclear_rate": unclear_rate,
        "pass": gate_pass,
        # old logit metrics kept for reference
        "logit_mean_p_coop": mean_p_coop,
        "logit_std_p_coop": std_p_coop,
    }

    (paths.RUN_DIR / "sanity.json").write_text(json.dumps(rec, indent=2))

    print(json.dumps(rec, indent=2))
    if gate_pass:
        print("GATE PASS")
    elif unclear_rate >= 0.25:
        print("GATE FAIL — unclear_rate too high (model not producing gradeable decisions)")
    elif coop_rate <= 0.1:
        print("GATE FAIL — coop_rate pinned low (model always defects; not content-responsive)")
    else:
        print("GATE FAIL — coop_rate pinned high (model always cooperates; not content-responsive)")


if __name__ == "__main__":
    main()
