"""Compare the fitted Trust vector to the fitted Cooperation vector.

After running ``run_full_pipeline.py`` on the trust corpus (output to
``pd_E4B_trust_v1`` or similar) and the existing PD pipeline (outputs in
``pd_full_v1``), this script answers the load-bearing question for the
trust experiment:

    *Is the Trust direction mechanistically distinct from the
    Cooperation direction, or is it the same direction with a different
    label?*

Operationalised as a per-layer cosine-similarity between the two unit-
normalised steering vectors. We also derive the model's TRUST CALIBRATION
CURVE by aggregating un-steered cooperation rate per payoff cell against
the cell's break-even betrayal probability.

Usage:
    python3 scripts/analyze_trust_vs_coop.py \\
        --coop-vectors local_data/pd_full_v1_dl/pd_full_v1/vectors.pt \\
        --trust-vectors local_data/pd_E4B_trust_v1_dl/pd_E4B_trust_v1/vectors.pt \\
        --trust-results local_data/pd_E4B_trust_v1_results/results_full.parquet \\
        --out-dir local_data/trust_vs_coop_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_vector_set(path: str):
    import torch
    from game_theory_llm.steering.storage import load_vector_set
    return load_vector_set(Path(path))


def cosine_similarity_per_layer(vs_coop, vs_trust, position: str = "mean_trace"):
    """Return dict {layer: cos_sim} for layers present in both sets."""
    import torch
    rows = []
    for (l, p), v_t in vs_trust.vectors.items():
        if p != position:
            continue
        key_c = (l, p)
        if key_c not in vs_coop.vectors:
            continue
        v_c = vs_coop.vectors[key_c]
        a = v_t.direction.float()
        b = v_c.direction.float()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
        rows.append({
            "layer": l, "position": p, "cosine": cos,
            "raw_norm_trust": v_t.raw_norm, "raw_norm_coop": v_c.raw_norm,
        })
    rows.sort(key=lambda r: r["layer"])
    return rows


def calibration_curve(results_parquet: str) -> list[dict]:
    """Group un-steered (α=0) trust rate by payoff cell and join breakeven_p.

    Returns one row per cell:
        {cell_id, breakeven_p, n_parsed, trust_rate, lower95, upper95}
    """
    import pandas as pd
    from scipy.stats import beta
    df = pd.read_parquet(results_parquet)
    df["parsed"] = df.decision.fillna("").astype(str).str.len() > 0
    # We only care about un-steered (α=0) rows for the natural calibration curve
    base = df[(df.alpha == 0.0) & df.parsed].copy()

    # The eval frames written by run_full_pipeline carry story_id; we need
    # cell_id, which comes from the trust corpus JSONL. Pull it out of story_id.
    base["cell_id"] = base.story_id.str.extract(r"^trust__([^_]+(?:_[^_]+)*)__")

    rows = []
    for cell_id, g in base.groupby("cell_id"):
        n = len(g)
        k = int(g.cooperated.sum())
        # 95% Wilson interval -> use beta exact for tiny n
        lo, hi = beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
        rows.append({"cell_id": cell_id, "n_parsed": n, "k_trust": k,
                     "trust_rate": k / max(1, n),
                     "lower95": float(lo), "upper95": float(hi)})

    # Join in breakeven_p from the static registry
    from game_theory_llm.trust_games import TRUST_GAMES
    cells_meta = {c.cell_id: c for c in TRUST_GAMES}
    for r in rows:
        c = cells_meta.get(r["cell_id"])
        r["breakeven_p"] = c.breakeven_p if c else None
        r["R"] = c.R if c else None
        r["P"] = c.P if c else None
        r["S"] = c.S if c else None

    rows.sort(key=lambda r: (r["breakeven_p"] if r["breakeven_p"] is not None else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coop-vectors", required=True,
                    help="vectors.pt fitted on the PD cooperation corpus")
    ap.add_argument("--trust-vectors", required=True,
                    help="vectors.pt fitted on the trust corpus")
    ap.add_argument("--trust-results", default=None,
                    help="results_full.parquet from the trust-corpus eval; "
                         "optional but lets us draw the calibration curve")
    ap.add_argument("--position", default="mean_trace")
    ap.add_argument("--out-dir", default="local_data/trust_vs_coop_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vs_coop = _load_vector_set(args.coop_vectors)
    vs_trust = _load_vector_set(args.trust_vectors)

    cos_rows = cosine_similarity_per_layer(vs_coop, vs_trust, position=args.position)
    cos_path = out_dir / "cosine_per_layer.json"
    cos_path.write_text(json.dumps(cos_rows, indent=2))

    print("\n=== Cosine similarity between Trust and Cooperation directions ===")
    print(f"{'layer':>6}  {'cos(trust, coop)':>18}  {'||trust||':>10}  {'||coop||':>10}")
    print("-" * 50)
    for r in cos_rows:
        print(f"{r['layer']:>6}  {r['cosine']:>18.3f}  "
              f"{r['raw_norm_trust']:>10.2f}  {r['raw_norm_coop']:>10.2f}")
    if cos_rows:
        cosines = [r["cosine"] for r in cos_rows]
        print(f"\nmean cosine: {np.mean(cosines):+.3f}   "
              f"max: {np.max(cosines):+.3f} at layer "
              f"{cos_rows[int(np.argmax(cosines))]['layer']}   "
              f"min: {np.min(cosines):+.3f} at layer "
              f"{cos_rows[int(np.argmin(cosines))]['layer']}")
        # Quick interpretation
        peak = max(cosines)
        if peak > 0.8:
            print("\n[interp] HIGH cosine at peak: trust-vector and coop-vector are largely the same direction.")
        elif peak > 0.5:
            print("\n[interp] MODERATE cosine: meaningfully overlapping but distinct directions.")
        else:
            print("\n[interp] LOW cosine: trust is mechanistically distinct from cooperation.")

    if args.trust_results:
        calib = calibration_curve(args.trust_results)
        calib_path = out_dir / "calibration_curve.json"
        calib_path.write_text(json.dumps(calib, indent=2))
        print("\n=== Trust calibration curve (α=0 baseline) ===")
        print(f"{'cell_id':<32}  {'p*':>5}  {'trust_rate':>10}  "
              f"{'95%CI':>16}  {'n':>4}")
        print("-" * 75)
        for r in calib:
            ci = f"[{r['lower95']:.2f}, {r['upper95']:.2f}]"
            ps = "  n/a" if r["breakeven_p"] is None else f"{r['breakeven_p']:5.2f}"
            print(f"{r['cell_id']:<32}  {ps}  "
                  f"{r['trust_rate']:>10.3f}  {ci:>16}  {r['n_parsed']:>4}")

    print(f"\nWrote analysis artefacts to {out_dir}")


if __name__ == "__main__":
    main()
