"""Tier-2 breadth readout: before/after per benchmark, per-op in-domain & extrapolation,
measurable-held-out transfer (>=3-benchmark rule), and the comparison vs Tier-1b depth-only.

Run after both eval suites: python3 scripts/analyze_tier2.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

D = Path("data/runs/gt_rlvr")
HELDOUT = ["boolean_eval", "dyck", "mmlu_pro", "bbh_hard"]   # all measurable at base (0.10-0.90)
TIER1B_TRANSFER = {"boolean_eval": +0.025, "dyck": -0.015, "mmlu_pro": +0.027}  # depth-only deltas


def load(tag, name):
    p = D / f"t2_{tag}_{name}.json"
    return json.load(open(p)) if p.exists() else None


def perop(d):
    out = defaultdict(list)
    for r in d["per_item"]:
        out[(r.get("family"), r.get("depth"))].append(r["correct"])
    return {k: sum(v) / len(v) for k, v in sorted(out.items())}, {k: len(v) for k, v in out.items()}


def main():
    names = ["indomain", "extrap"] + HELDOUT + ["gsm8k"]
    rows = {}
    print(f"{'benchmark':14s} {'base':>7} {'rlvr':>7} {'Δ':>8} | {'b_parse':>7} {'r_parse':>7}")
    for n in names:
        b, r = load("base", n), load("rlvr", n)
        if not (b and r):
            print(f"{n:14s} MISSING")
            continue
        d = r["accuracy"] - b["accuracy"]
        rows[n] = {"base": b["accuracy"], "rlvr": r["accuracy"], "delta": d,
                   "base_parse": b.get("parse_rate"), "rlvr_parse": r.get("parse_rate")}
        print(f"{n:14s} {b['accuracy']:7.3f} {r['accuracy']:7.3f} {d:+8.3f} | "
              f"{b.get('parse_rate', -1):7.3f} {r.get('parse_rate', -1):7.3f}")

    print("\n=== per-op breakdown (did BOTH ops learn? did either extrapolate?) ===")
    for n in ["indomain", "extrap"]:
        b, r = load("base", n), load("rlvr", n)
        if not (b and r):
            continue
        pb, nb = perop(b)
        pr, _ = perop(r)
        print(f"  [{n}]")
        for k in pb:
            print(f"    {k[0]:14s} d{k[1]}: base={pb[k]:.3f} rlvr={pr.get(k, float('nan')):.3f} "
                  f"Δ={pr.get(k, 0) - pb[k]:+.3f} (n={nb[k]})")

    xfer = {n: rows[n]["delta"] for n in HELDOUT if n in rows}
    if xfer:
        mean_x = sum(xfer.values()) / len(xfer)
        print(f"\n=== held-out transfer (>=3-benchmark rule) ===")
        for n, d in xfer.items():
            t1b = TIER1B_TRANSFER.get(n)
            print(f"  {n:14s} breadth Δ={d:+.3f}" + (f"   (Tier-1b depth-only Δ={t1b:+.3f})" if t1b is not None else ""))
        print(f"  mean breadth transfer = {mean_x:+.3f}  (Tier-1b depth-only mean = +0.012)")
    if "gsm8k" in rows:
        print(f"\n  gsm8k control Δ = {rows['gsm8k']['delta']:+.3f}")

    out = D / "tier2_breadth_analysis.json"
    out.write_text(json.dumps({"per_benchmark": rows,
                               "mean_heldout_transfer": (sum(xfer.values()) / len(xfer)) if xfer else None,
                               "tier1b_transfer": TIER1B_TRANSFER}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
