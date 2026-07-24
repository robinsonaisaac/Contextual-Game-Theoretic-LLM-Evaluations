#!/usr/bin/env python3
"""Analyze the Gemma-family behavioral battery (AAAI-27 revision).

Reads ``evals_gemma/`` (written by run_gemma_evals.py and the Modal E4B job),
computes per-game A-rates per Gemma model, compares against the paper's seven
frontier models (per-game ordering correlation), tests every
(gemma model, contrast dimension) framing contrast with chi-squared +
Bonferroni, and emits a LaTeX tabular block for the paper.

Usage:
    python3 scripts/analyze_gemma_evals.py [--run-id 2026-05-05-sharp]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game_theory_llm.analysis.loader import _load_evals, _load_stories

GAME_ORDER = ["harmony", "chicken", "prisoners_dilemma", "stag_hunt",
              "battle_of_the_sexes", "deadlock"]
GAME_LABEL = {
    "harmony": "Harmony", "chicken": "Chicken",
    "prisoners_dilemma": "Prisoner's Dilemma", "stag_hunt": "Stag Hunt",
    "battle_of_the_sexes": "Battle of the Sexes", "deadlock": "Deadlock",
}
# Frontier per-game A-rates as published (supplementary table); column order
# GPT, Opus, Sonnet, Gem Pro, Gem Flash, DeepSeek, Qwen.
FRONTIER_A = {
    "harmony":             [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "chicken":             [0.90, 0.84, 0.85, 0.89, 0.88, 0.87, 0.89],
    "prisoners_dilemma":   [0.84, 0.99, 0.98, 0.75, 0.85, 0.94, 0.94],
    "stag_hunt":           [0.48, 0.86, 0.90, 0.68, 0.87, 0.79, 0.80],
    "battle_of_the_sexes": [0.40, 0.42, 0.37, 0.37, 0.34, 0.35, 0.36],
    "deadlock":            [0.02, 0.03, 0.03, 0.03, 0.02, 0.03, 0.02],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-05-05-sharp")
    ap.add_argument("--root", default="data/runs")
    args = ap.parse_args()
    run_dir = Path(args.root) / args.run_id

    wide = _load_evals(run_dir / "evals_gemma", models=[])
    if wide.empty:
        sys.exit("no gemma evals found")
    models = sorted(c.replace("decision_", "") for c in wide.columns
                    if c.startswith("decision_"))
    long = wide.melt(id_vars=["cell_id", "story_idx"],
                     value_vars=[f"decision_{m}" for m in models],
                     var_name="model", value_name="decision")
    long["model"] = long["model"].str.replace("decision_", "", regex=False)
    long = long.dropna(subset=["decision"])  # rows scored so far
    long["game"] = long["cell_id"].str.split("__").str[0]
    long["dim"] = long["cell_id"].str.split("__").str[1]
    long["level"] = long["cell_id"].str.split("__").str[2]

    # Coverage + parse health.
    n_resp = wide[[f"response_{m}" for m in models]].notna().sum().sum()
    n_dec = len(long)
    print(f"models: {models}")
    print(f"responses on disk: {n_resp}; parseable decisions: {n_dec} "
          f"({100 * (1 - n_dec / max(n_resp, 1)):.1f}% unparseable)")

    # Per-game A-rates.
    arate = (long.assign(is_a=lambda d: d.decision.eq("A"))
             .groupby(["game", "model"])["is_a"].agg(["mean", "count"])
             .reset_index())
    piv = arate.pivot(index="game", columns="model", values="mean").reindex(GAME_ORDER)
    cnt = arate.pivot(index="game", columns="model", values="count").reindex(GAME_ORDER)
    print("\nPer-game A-rates:")
    print(piv.round(2).to_string())
    print("\nCounts:")
    print(cnt.to_string())
    overall = long.assign(is_a=lambda d: d.decision.eq("A")).groupby("model")["is_a"].mean()
    print("\nOverall A-rate:", overall.round(2).to_dict())

    # Ordering correlation vs frontier mean.
    frontier_mean = {g: float(np.mean(v)) for g, v in FRONTIER_A.items()}
    print("\nSpearman rho of per-game A-rate vs frontier mean:")
    for m in models:
        v = piv[m].dropna()
        games = [g for g in GAME_ORDER if g in v.index]
        rho, p = sps.spearmanr([frontier_mean[g] for g in games], v.loc[games])
        r, _ = sps.pearsonr([frontier_mean[g] for g in games], v.loc[games])
        print(f"  {m}: rho={rho:.3f} (p={p:.3g}), pearson r={r:.3f}, n_games={len(games)}")

    # Framing contrasts: chi-squared per (model, dim), Bonferroni over tests.
    print("\nFraming contrasts (chi-squared, Bonferroni-corrected):")
    tests = []
    for m in models:
        for dim in sorted(long.dim.unique()):
            sub = long[(long.model == m) & (long.dim == dim)]
            lv = sorted(sub.level.unique())
            if len(lv) < 2:
                continue
            tab = np.array([
                [int(((sub.level == l) & (sub.decision == d)).sum())
                 for d in ("A", "B")] for l in lv[:2]])
            if tab.sum() == 0 or (tab.sum(axis=1) == 0).any():
                continue
            chi2, p, _, _ = sps.chi2_contingency(tab)
            d0 = tab[0][0] / max(tab[0].sum(), 1)
            d1 = tab[1][0] / max(tab[1].sum(), 1)
            tests.append((m, dim, p, d1 - d0))
    n_tests = len(tests)
    sig = [(m, d, p * n_tests, delta) for m, d, p, delta in tests if p * n_tests < 0.05]
    for m, d, p, delta in tests:
        mark = " *SIG*" if p * n_tests < 0.05 else ""
        print(f"  {m} x {d}: p_corr={min(p * n_tests, 1):.3g} delta={delta:+.3f}{mark}")
    print(f"\n{len(sig)}/{n_tests} (model, dim) contrasts survive Bonferroni.")

    # LaTeX block.
    print("\n--- LaTeX rows (game x gemma models) ---")
    hdr = " & ".join(m.replace("gemma-", "").replace("-it", "") for m in models)
    print(f"% columns: Game & {hdr}")
    for g in GAME_ORDER:
        cells = " & ".join(
            f"{piv.loc[g, m]:.2f}" if g in piv.index and pd.notna(piv.loc[g, m]) else "--"
            for m in models)
        print(f"{GAME_LABEL[g]} & {cells} \\\\")
    cells = " & ".join(f"{overall[m]:.2f}" for m in models)
    print(f"\\midrule\n\\textbf{{Overall}} & {cells} \\\\")

    out = run_dir / "evals_gemma" / "_analysis_summary.json"
    out.write_text(json.dumps({
        "models": models,
        "a_rates": {g: {m: (None if pd.isna(piv.loc[g, m]) else round(float(piv.loc[g, m]), 4))
                        for m in models} for g in GAME_ORDER if g in piv.index},
        "overall": {m: round(float(overall[m]), 4) for m in models},
        "n_significant_contrasts": len(sig), "n_contrast_tests": n_tests,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
