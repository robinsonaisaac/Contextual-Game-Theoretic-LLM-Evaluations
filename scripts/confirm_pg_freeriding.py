#!/usr/bin/env python3
"""LOCKED confirmatory analysis for docs/preregistration_pg_freeriding.md.

Committed BEFORE the confirmatory data exists. Run exactly once, on the
complete run. It deliberately has no options for choosing a different DV, test,
or subset — the point of the exercise is that none of those choices remain open
once the data lands.

    python3 scripts/confirm_pg_freeriding.py data/runs/pg_confirm_v1

Refuses to run on an incomplete run unless --force-incomplete is passed, which
exists only so a genuinely truncated run can be reported honestly as such
rather than silently analysed as if it were complete.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

N_REQUIRED = 30
CELLS = ("baseline", "k5_a-4", "k5_a+4")


def load(run: Path):
    rows = json.loads((run / "manifest.json").read_text())
    out = {}
    for c in CELLS:
        rs = [r for r in rows if r["label"] == c and r.get("winner")
              and r.get("metrics")]
        out[c] = {
            "free_ride": [r["metrics"]["free_ride_rate"] for r in rs
                          if r["metrics"].get("free_ride_rate") is not None],
            "mean_contrib": [r["metrics"]["mean_contribution_rate"] for r in rs
                             if r["metrics"].get("mean_contribution_rate") is not None],
        }
    return out


def mw(a, b):
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        val = max(prev, min(1.0, (m - i) * p))
        adj[k] = val
        prev = val
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--force-incomplete", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    d = load(run)

    print("=" * 72)
    print("CONFIRMATORY ANALYSIS — docs/preregistration_pg_freeriding.md")
    print("=" * 72)
    ns = {c: len(d[c]["free_ride"]) for c in CELLS}
    print("completed matches per cell:", ns)
    short = [c for c in CELLS if ns[c] < N_REQUIRED]
    if short and not args.force_incomplete:
        print(f"\nREFUSING TO RUN: {short} below the pre-registered n={N_REQUIRED}.")
        print("Top up with additional seeds from 1030, or pass --force-incomplete")
        print("to report an explicitly truncated run.")
        return 1
    if short:
        print(f"\n*** TRUNCATED RUN: {short} below n={N_REQUIRED}. "
              "Results are underpowered relative to the pre-registration. ***")

    fr = {c: d[c]["free_ride"] for c in CELLS}
    mc = {c: d[c]["mean_contrib"] for c in CELLS}
    for c in CELLS:
        print(f"  {c:9s} free-ride mean={st.mean(fr[c]):.4f} "
              f"median={st.median(fr[c]):.4f} | "
              f"mean-contrib mean={st.mean(mc[c]):.4f}")

    # ---- PRIMARY --------------------------------------------------------
    p1 = mw(fr["k5_a-4"], fr["k5_a+4"])
    U = stats.mannwhitneyu(fr["k5_a-4"], fr["k5_a+4"],
                           alternative="two-sided").statistic
    rb = abs(1 - 2 * U / (len(fr["k5_a-4"]) * len(fr["k5_a+4"])))
    print("\n--- PRIMARY (H1): free-ride rate, alpha=-4 vs alpha=+4 ---")
    print(f"  {st.mean(fr['k5_a-4']):.4f} vs {st.mean(fr['k5_a+4']):.4f}   "
          f"Mann-Whitney p = {p1:.5f}   rank-biserial r = {rb:.3f}")
    print(f"  VERDICT: {'CONFIRMED' if p1 < 0.05 else 'NOT CONFIRMED'} at alpha=0.05")

    # ---- SECONDARY (Holm) ----------------------------------------------
    raw = {
        "H2 a-4 vs baseline (free-ride)": mw(fr["k5_a-4"], fr["baseline"]),
        "H3 a+4 vs baseline (free-ride)": mw(fr["k5_a+4"], fr["baseline"]),
        "H4 a-4 vs a+4 (mean contribution)": mw(mc["k5_a-4"], mc["k5_a+4"]),
    }
    adj = holm(raw)
    pred = {
        "H2 a-4 vs baseline (free-ride)": "significant",
        "H3 a+4 vs baseline (free-ride)": "NOT significant",
        "H4 a-4 vs a+4 (mean contribution)": "NOT significant",
    }
    print("\n--- SECONDARY (Holm-corrected across 3) ---")
    for k in raw:
        sig = adj[k] < 0.05
        got = "significant" if sig else "not significant"
        ok = "as predicted" if got.lower().startswith(pred[k].lower()[:3]) else "AGAINST PREDICTION"
        print(f"  {k:38s} raw p={raw[k]:.5f}  Holm p={adj[k]:.5f}  "
              f"-> {got:15s} ({pred[k]} predicted: {ok})")

    print("\n--- ONE-SIDEDNESS CLAIM ---")
    one_sided = (p1 < 0.05
                 and adj["H2 a-4 vs baseline (free-ride)"] < 0.05
                 and adj["H3 a+4 vs baseline (free-ride)"] >= 0.05)
    print("  'removes cooperation but does not install it':",
          "SUPPORTED" if one_sided else "NOT SUPPORTED")
    if not one_sided:
        print("  (requires H1 significant, H2 significant, H3 non-significant)")

    (run / "confirmatory_result.json").write_text(json.dumps({
        "n": ns, "primary_p": p1, "rank_biserial": rb,
        "secondary_raw": raw, "secondary_holm": adj,
        "one_sidedness_supported": bool(one_sided),
        "free_ride_means": {c: st.mean(fr[c]) for c in CELLS},
        "mean_contrib_means": {c: st.mean(mc[c]) for c in CELLS},
    }, indent=2))
    print(f"\nwrote {run}/confirmatory_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
