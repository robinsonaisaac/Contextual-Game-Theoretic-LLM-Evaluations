#!/usr/bin/env python3
"""Per-condition stats for a steering-in-games run: mean, 95% CI, and
Mann-Whitney U p-value vs the shared baseline for each judge index, plus the
cooperative-side win rate. Reads results/per_match.json (written by
analyze_game_steering.py). Used to populate the report + paper tables so ONW
and Secret Hitler get identical statistical treatment.

Usage:
    python3 scripts/steering_games_stats.py --results data/runs/sh_steering_v1/results
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

METRICS = ["cooperation_index", "trust_index", "aggression_index",
           "n_trades_proposed", "n_trades_completed", "n_promises", "renege_rate"]
ORDER = ["coop_a-4", "baseline", "coop_a+4", "trust_a-4", "trust_a+4"]


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rows = json.loads((Path(args.results) / "per_match.json").read_text())
    # Task 2's unparseable-output void (`aborted`): a match that never
    # reached a terminal state has no meaningful judge/behaviour metrics and
    # must not dilute any stat below, same exclusion `aggregate()` in
    # analyze_game_steering.py applies.
    rows = [r for r in rows if not r.get("aborted")]

    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    base = by.get("baseline", [])

    def col(rs, k):
        return [r.get(k) for r in rs if r.get(k) is not None]

    labels = [l for l in ORDER if l in by] + [l for l in by if l not in ORDER]
    print(f"\n=== {args.results} (baseline n={len(base)}) ===")
    hdr = f"{'condition':12s} {'n':>3} | " + " | ".join(
        f"{m.split('_')[0]:>22s}" for m in METRICS) + f" | {'coopWin':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for lab in labels:
        rs = by[lab]
        cells = []
        for m in METRICS:
            mu, ci, n = mean_ci(col(rs, m))
            base_mu, _, _ = mean_ci(col(base, m))
            # Several of the newer metrics are legitimately None for an
            # entire run -- n_trades_* on any non-Monopoly game, n_promises/
            # renege_rate when the run used --no-judge/--no-promises or every
            # judge reply was malformed. Unlike cooperation/trust/aggression
            # (always present when judged), that must render as a placeholder
            # rather than crash on `f"{None:.1f}"`.
            if mu is None:
                cells.append(f"{'--':>5s} (Δ{'--':>5s},{'    -   '})")
                continue
            if lab == "baseline" or not base or base_mu is None:
                p = None
            else:
                a, b = col(rs, m), col(base, m)
                p = mannwhitneyu(a, b, alternative="two-sided").pvalue if a and b else None
            d = (mu - base_mu) if (base_mu is not None and lab != "baseline") else 0.0
            ptxt = f"p={p:.3g}" if p is not None else "    -   "
            cells.append(f"{mu:5.1f} (Δ{d:+5.1f},{ptxt})")
        win, _, _ = mean_ci(col(rs, "coop_side_win"))
        print(f"{lab:12s} {len(rs):>3} | " + " | ".join(cells) + f" | {win:8.2f}")
    print()


if __name__ == "__main__":
    main()
