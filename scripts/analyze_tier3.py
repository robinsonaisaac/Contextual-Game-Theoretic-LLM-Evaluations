"""Tier-3 readout: process-reward vs outcome-reward, three-arm comparison.

PRIMARY: d6 extrapolation (base vs outcome vs process), two-proportion z-test on
process-vs-outcome. SECONDARY: in-domain per depth, held-out transfer (base numbers
from t2_base_* at matched budgets), gsm8k control, training dead-group rates.

Run after both eval suites: python3 scripts/analyze_tier3.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

D = Path("data/runs/gt_rlvr")
ARMS = ["base", "outcome", "process"]
TRANSFER = ["boolean_eval", "dyck", "mmlu_pro", "bbh_hard"]


def load(arm, name):
    # base transfer/control numbers come from the Tier-2 suite (identical budgets)
    for pref in ([f"t3_{arm}_"] if arm != "base" else [f"t3_{arm}_", f"t2_{arm}_"]):
        p = D / f"{pref}{name}.json"
        if p.exists():
            return json.load(open(p))
    return None


def ztest(p1, n1, p2, n2):
    """Two-proportion z-test; returns (z, two-sided p)."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-9
    z = (p1 - p2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main():
    print("=== PRIMARY: d6 extrapolation @8192 (n=160) ===")
    d6 = {}
    for arm in ARMS:
        d = load(arm, "d6")
        if d:
            d6[arm] = d
            print(f"  {arm:8s} acc={d['accuracy']:.3f} parse={d.get('parse_rate', -1):.3f}")
    if "outcome" in d6 and "process" in d6:
        n = len(d6["process"]["per_item"])
        z, p = ztest(d6["process"]["accuracy"], n, d6["outcome"]["accuracy"], n)
        print(f"  process vs outcome: Δ={d6['process']['accuracy'] - d6['outcome']['accuracy']:+.3f} "
              f"(z={z:+.2f}, p={p:.3f})")
        if "base" in d6:
            zb, pb = ztest(d6["process"]["accuracy"], n, d6["base"]["accuracy"],
                           len(d6["base"]["per_item"]))
            print(f"  process vs base   : Δ={d6['process']['accuracy'] - d6['base']['accuracy']:+.3f} "
                  f"(z={zb:+.2f}, p={pb:.3f})")

    print("\n=== in-domain per depth ===")
    perd = {}
    for arm in ARMS:
        d = load(arm, "indomain")
        if not d:
            continue
        by = defaultdict(list)
        for r in d["per_item"]:
            by[r.get("depth")].append(r["correct"])
        perd[arm] = {k: sum(v) / len(v) for k, v in by.items()}
    for dep in sorted(next(iter(perd.values()), {})):
        line = f"  d{dep}: " + "  ".join(f"{arm}={perd[arm].get(dep, float('nan')):.3f}"
                                          for arm in ARMS if arm in perd)
        print(line)

    print("\n=== held-out transfer (vs base from matched-budget t2 suite) ===")
    rows = {}
    for name in TRANSFER + ["gsm8k"]:
        vals = {arm: load(arm, name) for arm in ARMS}
        if not all(vals.get(a) for a in ARMS):
            print(f"  {name:14s} MISSING {[a for a in ARMS if not vals.get(a)]}")
            continue
        b, o, pr = (vals[a]["accuracy"] for a in ARMS)
        rows[name] = {"base": b, "outcome": o, "process": pr}
        tag = "CONTROL" if name == "gsm8k" else ""
        print(f"  {name:14s} base={b:.3f} outcome={o:.3f} ({o - b:+.3f})  "
              f"process={pr:.3f} ({pr - b:+.3f})  {tag}")
    xfer_o = [rows[n]["outcome"] - rows[n]["base"] for n in TRANSFER if n in rows]
    xfer_p = [rows[n]["process"] - rows[n]["base"] for n in TRANSFER if n in rows]
    if xfer_o:
        print(f"  mean transfer: outcome={sum(xfer_o)/len(xfer_o):+.3f}  "
              f"process={sum(xfer_p)/len(xfer_p):+.3f}")

    print("\n=== training health (dead groups at hardest tier, batches>=26) ===")
    for arm in ["outcome", "process"]:
        p = D / f"tier3_{arm}_30b/metrics.jsonl"
        if p.exists():
            recs = [json.loads(l) for l in open(p) if l.strip()]
            tail = [r for r in recs if r.get("progress/batch", 0) >= 26]
            if tail:
                ab = sum(r.get("env/all/by_group/frac_all_bad", 0) for r in tail) / len(tail)
                mx = sum(r.get("env/all/by_group/frac_mixed", 0) for r in tail) / len(tail)
                print(f"  {arm:8s} mean all_bad={ab:.2f} mixed={mx:.2f}")

    out = D / "tier3_analysis.json"
    out.write_text(json.dumps({"d6": {a: d6[a]["accuracy"] for a in d6},
                               "indomain": {a: perd.get(a) for a in perd},
                               "transfer": rows}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
