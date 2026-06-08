"""Operation-overlap analysis (Tier-1 decisive readout).

Joins base vs RLVR per-item results across all eval benchmarks and tests the three
hypotheses:
  H_op  (islands)     : improvement rises with trained-operation overlap  -> ov slope > 0
  H_gen (discipline)  : roughly uniform improvement regardless of overlap -> flat slope, lift>0
  H_scale/null        : no improvement

Measurability: an item's (benchmark,depth) cell must have base accuracy in [lo,hi]; saturated
cells (ceiling/floor) carry no transfer signal and are excluded (per-depth, post-hoc).

Run: python3 scripts/analyze_overlap.py            (py3.9; pandas + statsmodels)
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from game_theory_llm.reasoning.op_taxonomy import overlap, TRAINED_TAGS

D = "data/runs/gt_rlvr"
LO, HI = 0.10, 0.90        # measurable base-accuracy window per (benchmark,depth) cell


def load(tag):
    rows = []
    for p in glob.glob(f"{D}/t1_{tag}_*.json"):
        bench = Path(p).stem.replace(f"t1_{tag}_", "")
        for r in json.load(open(p))["per_item"]:
            r["benchmark"] = bench
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    b, f = load("base"), load("rlvr")
    m = b.merge(f, on="story_id", suffixes=("_b", "_f"))
    m["benchmark"] = m["benchmark_b"]
    m["depth"] = m["depth_b"].fillna(-1)
    m["ov"] = m["op_tags_b"].apply(lambda t: overlap(t or [], TRAINED_TAGS))
    m["know"] = m["knowledge_b"].fillna(False)
    m["dimp"] = m["correct_f"].astype(int) - m["correct_b"].astype(int)

    # per (benchmark,depth) base accuracy -> measurability mask
    cell = m.groupby(["benchmark", "depth"])["correct_b"].mean().rename("base_cell")
    m = m.merge(cell, on=["benchmark", "depth"])
    meas = m[(m["base_cell"] >= LO) & (m["base_cell"] <= HI)].copy()

    print(f"items total={len(m)} measurable={len(meas)} "
          f"({meas['benchmark'].nunique()} benchmarks)")

    print("\n=== per-benchmark (measurable cells) base -> rlvr ===")
    for bench, g in meas.groupby("benchmark"):
        ba = g["correct_b"].mean(); fa = g["correct_f"].mean()
        print(f"  {bench:16s} n={len(g):>4} base={ba:.3f} rlvr={fa:.3f} d={fa-ba:+.3f} "
              f"ov={g['ov'].mean():.2f} know={bool(g['know'].iloc[0])}")

    print("\n=== operation-overlap: improvement by trained-overlap bin ===")
    for lo, hi, lab in [(-0.01, 0.01, "ov=0 (held-out op)"),
                        (0.01, 0.99, "0<ov<1 (mixed)"),
                        (0.99, 1.01, "ov=1 (trained op)")]:
        g = meas[(meas["ov"] > lo) & (meas["ov"] <= hi)] if lo >= 0 else meas[meas["ov"] <= hi]
        if len(g):
            print(f"  {lab:22s} n={len(g):>4} mean improvement={g['dimp'].mean():+.3f}")

    # regression: dimp ~ ov + depth + C(benchmark)
    try:
        import statsmodels.formula.api as smf
        if meas["ov"].nunique() > 1:
            mod = smf.ols("dimp ~ ov + depth + C(benchmark)", data=meas).fit()
            print(f"\n=== overlap slope (primary): {mod.params['ov']:+.3f} "
                  f"(p={mod.pvalues['ov']:.3f})  [H_op if >0 & sig] ===")
            print(f"    intercept lift (H_gen if >0): {mod.params['Intercept']:+.3f}")
    except Exception as e:
        print("regression skipped:", e)

    print("\n=== knowledge-free vs knowledge-heavy (dissociation) ===")
    for k, g in meas.groupby("know"):
        print(f"  {'knowledge-heavy' if k else 'knowledge-free':16s} n={len(g):>4} "
              f"mean improvement={g['dimp'].mean():+.3f}")

    print("\n=== in-domain depth-extrapolation (replication anchor) ===")
    de = m[m["benchmark"] == "depth_extrap"]
    for d, g in de.groupby("depth"):
        print(f"  depth {int(d)} n={len(g):>3} base={g['correct_b'].mean():.3f} "
              f"rlvr={g['correct_f'].mean():.3f} d={g['correct_f'].mean()-g['correct_b'].mean():+.3f}")

    out = Path(f"{D}/overlap_analysis.json")
    out.write_text(json.dumps({
        "n_measurable": int(len(meas)),
        "per_benchmark": {bm: {"base": float(g["correct_b"].mean()),
                               "rlvr": float(g["correct_f"].mean()),
                               "delta": float(g["correct_f"].mean() - g["correct_b"].mean()),
                               "ov": float(g["ov"].mean())}
                          for bm, g in meas.groupby("benchmark")},
    }, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
