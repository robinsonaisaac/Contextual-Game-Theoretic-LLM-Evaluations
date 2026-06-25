"""Causal / mediation test: does dilemma-RECOGNITION causally drive cooperation BEHAVIOR?

Clamps the recognition direction in the residual stream at layer 24 during generation,
measures cooperate-rate via Sonnet judge, with a random-feature control and dose-response.

Stages:
  calibrate — smoke run on first ~12 scenarios (baseline, recog k=-2, recog k=+2)
               prints cooperate-rates, unclear-rates, sample continuations, and
               recommends whether the k∈{-2,-1,0,1,2}·M range needs widening.
  sweep      — full dose-response: conditions {recog, random} × k∈{-2,-1,0,1,2}
               writes data/runs/saemap_9b/causal/sweep.json + verdict.json

Usage:
    cd /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering
    source ../../.env          # or let dotenv load it
    .venv-sae/bin/python scripts/saemap_causal.py --stage calibrate
    .venv-sae/bin/python scripts/saemap_causal.py --stage sweep      # launched by controller
"""
from __future__ import annotations
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(
    "/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env"
)

from game_theory_llm.saemap import corpus, judge, paths, remote
from game_theory_llm.saemap.sae import QwenScopeSAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAYER = 24
REC_FEATURES = [48983, 29366, 14407, 51436, 16610]
# Five hash-seeded random feature indices for the control direction (L24, D_SAE=65536).
# Seeds chosen arbitrarily and fixed so rand_dir is reproducible.
_RAND_SEEDS = [0x1F2E3D4C, 0x5B6A7980, 0xABCDEF01, 0x11223344, 0x99887766]
CAUSAL_DIR = paths.RUN_DIR / "causal"
PARTIAL_PATH = CAUSAL_DIR / "sweep_partial.jsonl"


# ---------------------------------------------------------------------------
# Direction construction
# ---------------------------------------------------------------------------
def build_rec_dir(sae: QwenScopeSAE) -> torch.Tensor:
    """Unit steering direction = unit( sum_f W_dec[:, f] ) over REC_FEATURES."""
    vecs = [sae.W_dec[:, f] for f in REC_FEATURES]
    s = sum(vecs)
    return s / s.norm()


def build_rand_dir(sae: QwenScopeSAE) -> torch.Tensor:
    """Matched control: unit( sum of 5 hash-seeded random decoder columns at L24 ).

    Construction mirrors rec_dir: pick 5 feature indices via seeded RNG,
    sum their decoder columns, normalise. This controls for the fact that
    any multi-feature sum direction has unit norm and ~similar geometry.
    """
    rng = np.random.default_rng(0xDEADBEEF)
    D_SAE = sae.W_dec.shape[1]
    # Pick 5 distinct random indices using the fixed seeds mixed into one RNG seed
    combined_seed = sum(_RAND_SEEDS) & 0xFFFFFFFF
    rng2 = np.random.default_rng(combined_seed)
    rand_feats = rng2.choice(D_SAE, size=5, replace=False).tolist()
    vecs = [sae.W_dec[:, f] for f in rand_feats]
    s = sum(vecs)
    return s / s.norm()


def compute_M(rec_dir: torch.Tensor) -> float:
    """M = (dilemma_mean - nondilemma_mean)_L24 · rec_dir.

    This is the natural recognition signal magnitude in rec_dir units.
    """
    raw_d = torch.load(
        paths.RUN_DIR / "activations/raw/recog_dilemma.pt",
        map_location="cpu", weights_only=False,
    )
    raw_nd = torch.load(
        paths.RUN_DIR / "activations/raw/recog_nondilemma.pt",
        map_location="cpu", weights_only=False,
    )
    layers: list = raw_d["layers"]
    l24_idx = layers.index(LAYER)
    dil_mean = raw_d["residuals"][:, l24_idx, :].float().mean(0)
    nondil_mean = raw_nd["residuals"][:, l24_idx, :].float().mean(0)
    M = float((dil_mean - nondil_mean) @ rec_dir)
    return M


# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------
def eval_condition(
    rows: list[dict],
    vec: np.ndarray,
    condition_label: str,
    k: float,
) -> dict:
    """Generate continuations with the clamped vector, judge them, return rates."""
    elic_prompts = [corpus.decision_elicitation(r["prompt"]) for r in rows]

    if k == 0.0:
        # Baseline: use plain generate (no hook), vec is zero-norm anyway,
        # but using generate avoids modal serialization of a zero vector.
        continuations = remote.generate(
            elic_prompts, max_new_tokens=2000, temperature=0.0,
            stop_string="</decision>",
        )
    else:
        continuations = remote.causal_generate(
            elic_prompts, layer=LAYER, vec=vec.tolist(),
            max_new_tokens=2000, temperature=0.0, seed=0,
            stop_string="</decision>",
        )

    items = [
        {
            "scenario": r["prompt"],
            "continuation": c,
            "coop_letter": r["coop_letter"],
        }
        for r, c in zip(rows, continuations)
    ]
    verdicts = judge.grade_decisions(items)

    n_coop = sum(1 for v in verdicts if v["verdict"] == "cooperate")
    n_defect = sum(1 for v in verdicts if v["verdict"] == "defect")
    n_unclear = sum(1 for v in verdicts if v["verdict"] == "unclear")
    n_gradeable = n_coop + n_defect
    coop_rate = n_coop / max(1, n_gradeable)
    unclear_rate = n_unclear / len(rows)

    return {
        "condition": condition_label,
        "k": k,
        "n": len(rows),
        "n_coop": n_coop,
        "n_defect": n_defect,
        "n_unclear": n_unclear,
        "coop_rate": coop_rate,
        "unclear_rate": unclear_rate,
        "continuations": continuations,
        "verdicts": [v["verdict"] for v in verdicts],
    }


# ---------------------------------------------------------------------------
# Calibrate stage
# ---------------------------------------------------------------------------
def run_calibrate(M: float, rec_dir: np.ndarray) -> None:
    """Smoke run: baseline + recog k=±2 on first 12 scenarios."""
    rows = corpus.pd_eval_set(limit=12)
    print(f"[calibrate] M = {M:.4f}  (dilemma−nondilemma projection onto rec_dir at L{LAYER})")
    print(f"[calibrate] running on {len(rows)} scenarios; conditions: baseline, recog k=-2, recog k=+2")
    print()

    results = {}

    # Baseline (k=0)
    print("[calibrate] --- baseline (k=0) ---")
    zero_vec = np.zeros(len(rec_dir), dtype=np.float32)
    res_base = eval_condition(rows, zero_vec, "recog", 0.0)
    results["baseline"] = res_base
    print(f"  coop_rate={res_base['coop_rate']:.3f}  unclear_rate={res_base['unclear_rate']:.3f}  "
          f"(n_coop={res_base['n_coop']} n_defect={res_base['n_defect']} n_unclear={res_base['n_unclear']})")

    # recog k=+2
    print("[calibrate] --- recog k=+2 ---")
    vec_p2 = (2.0 * M * rec_dir).astype(np.float32)
    res_p2 = eval_condition(rows, vec_p2, "recog", +2.0)
    results["recog_k+2"] = res_p2
    print(f"  coop_rate={res_p2['coop_rate']:.3f}  unclear_rate={res_p2['unclear_rate']:.3f}  "
          f"(n_coop={res_p2['n_coop']} n_defect={res_p2['n_defect']} n_unclear={res_p2['n_unclear']})")

    # recog k=-2
    print("[calibrate] --- recog k=-2 ---")
    vec_m2 = (-2.0 * M * rec_dir).astype(np.float32)
    res_m2 = eval_condition(rows, vec_m2, "recog", -2.0)
    results["recog_k-2"] = res_m2
    print(f"  coop_rate={res_m2['coop_rate']:.3f}  unclear_rate={res_m2['unclear_rate']:.3f}  "
          f"(n_coop={res_m2['n_coop']} n_defect={res_m2['n_defect']} n_unclear={res_m2['n_unclear']})")

    print()
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  M (natural recognition signal) = {M:.4f}")
    print(f"  baseline  coop_rate = {res_base['coop_rate']:.3f}  unclear_rate = {res_base['unclear_rate']:.3f}")
    print(f"  recog k=-2  coop_rate = {res_m2['coop_rate']:.3f}  unclear_rate = {res_m2['unclear_rate']:.3f}")
    print(f"  recog k=+2  coop_rate = {res_p2['coop_rate']:.3f}  unclear_rate = {res_p2['unclear_rate']:.3f}")
    delta = res_p2["coop_rate"] - res_m2["coop_rate"]
    print(f"  Δcoop_rate (k+2 − k-2) = {delta:+.3f}")
    print()

    # Sample continuations (2 per condition, truncated to 400 chars)
    print("SAMPLE CONTINUATIONS (2 per condition, truncated to 400 chars)")
    print("-" * 70)
    for label, res in [("baseline (k=0)", res_base),
                        ("recog k=+2", res_p2),
                        ("recog k=-2", res_m2)]:
        print(f"\n  [{label}]")
        for i in range(min(2, len(res["continuations"]))):
            trunc = res["continuations"][i][:400].replace("\n", " ").strip()
            verdict = res["verdicts"][i]
            row_id = rows[i]["id"]
            print(f"    [{i+1}] {row_id} | verdict={verdict}")
            print(f"         {trunc}{'...' if len(res['continuations'][i]) > 400 else ''}")

    print()
    print("=" * 70)
    print("ALPHA RANGE RECOMMENDATION")
    print("=" * 70)
    if abs(delta) < 0.10:
        print(f"  Δcoop_rate={delta:+.3f} is SMALL (<0.10) at k=±2.")
        print("  RECOMMENDATION: widen sweep to k∈{-4,-2,-1,0,1,2,4} — the ±2M signal")
        print("  may be below the model's behavioral threshold for this direction.")
        print("  Consider also checking unclear_rate at higher k to verify the direction")
        print("  isn't causing incoherent outputs rather than redirected choices.")
    elif abs(delta) > 0.50:
        print(f"  Δcoop_rate={delta:+.3f} is LARGE (>0.50) at k=±2.")
        print("  RECOMMENDATION: k∈{-2,-1,0,1,2}·M range is valid but consider also")
        print("  adding k=±0.5 or ±1 for finer dose-response resolution.")
        if res_p2["unclear_rate"] > 0.25 or res_m2["unclear_rate"] > 0.25:
            print("  WARNING: unclear_rate elevated at k=±2 — strong injection may")
            print("  be disrupting coherence; verify sample texts above.")
    else:
        print(f"  Δcoop_rate={delta:+.3f} — k∈{{-2,-1,0,1,2}}·M range is SENSIBLE.")
        print("  RECOMMENDATION: proceed with the full sweep at k∈{-2,-1,0,1,2}.")
    print("=" * 70)

    # Save calibration results
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    cal_out = {
        "M": M,
        "layer": LAYER,
        "rec_features": REC_FEATURES,
        "results": {k: {kk: v for kk, v in vv.items() if kk != "continuations"}
                    for k, vv in results.items()},
        "sample_continuations": {
            label: res["continuations"][:2]
            for label, res in [("baseline", res_base), ("recog_kp2", res_p2),
                                ("recog_km2", res_m2)]
        },
    }
    (CAUSAL_DIR / "calibrate.json").write_text(json.dumps(cal_out, indent=2))
    print(f"\n[calibrate] results saved to {CAUSAL_DIR / 'calibrate.json'}")


# ---------------------------------------------------------------------------
# Partial save / resume helpers
# ---------------------------------------------------------------------------

def _load_partial() -> dict[tuple[str, float], dict]:
    """Read sweep_partial.jsonl; return {(condition, k): row_dict}."""
    if not PARTIAL_PATH.exists():
        return {}
    done: dict[tuple[str, float], dict] = {}
    with PARTIAL_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["condition"], float(row["k"]))
                done[key] = row
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append_partial(row: dict) -> None:
    """Append one completed cell to sweep_partial.jsonl (creates parent dir if needed)."""
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    with PARTIAL_PATH.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Sweep stage
# ---------------------------------------------------------------------------
def run_sweep(M: float, rec_dir: np.ndarray, rand_dir: np.ndarray) -> None:
    """Full dose-response sweep: {recog, random} × k∈{-8,-4,-2,-1,0,1,2,4,8}.

    Supports incremental save/resume: completed cells are written to
    sweep_partial.jsonl after each call; on restart, already-done cells are
    skipped.
    """
    rows = corpus.pd_eval_set(limit=50)
    print(f"[sweep] M={M:.4f}  n_scenarios={len(rows)}")
    print(f"[sweep] conditions: recog, random  |  k∈{{-8,-4,-2,-1,0,1,2,4,8}}")

    # --- Resume: load previously completed cells ---
    done = _load_partial()
    if done:
        print(f"[sweep] resuming — skipping {len(done)} completed cells")

    k_vals = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
    zero_vec = np.zeros(len(rec_dir), dtype=np.float32)

    sweep_results: list[dict] = []
    recog_rates: dict[float, float] = {}
    random_rates: dict[float, float] = {}

    # Populate from already-done cells
    for (cond, k), row in done.items():
        sweep_results.append(row)
        if cond == "baseline":
            recog_rates[0.0] = row["coop_rate"]
            random_rates[0.0] = row["coop_rate"]
        elif cond == "recog":
            recog_rates[k] = row["coop_rate"]
        elif cond == "random":
            random_rates[k] = row["coop_rate"]

    # --- Baseline (k=0), shared between both conditions ---
    baseline_key = ("baseline", 0.0)
    if baseline_key not in done:
        print(f"\n[sweep] baseline (k=0)...")
        res_base = eval_condition(rows, zero_vec, "recog", 0.0)
        base_row = {
            "condition": "baseline",
            "k": 0.0,
            "coop_rate": res_base["coop_rate"],
            "unclear_rate": res_base["unclear_rate"],
            "n_coop": res_base["n_coop"],
            "n_defect": res_base["n_defect"],
            "n_unclear": res_base["n_unclear"],
        }
        sweep_results.append(base_row)
        _append_partial(base_row)
        recog_rates[0.0] = res_base["coop_rate"]
        random_rates[0.0] = res_base["coop_rate"]
        print(f"  coop_rate={res_base['coop_rate']:.3f}  unclear_rate={res_base['unclear_rate']:.3f}")
    else:
        print(f"[sweep] baseline (k=0) — already done, skipping")

    for cond, direction, rate_dict in [
        ("recog", rec_dir, recog_rates),
        ("random", rand_dir, random_rates),
    ]:
        for k in [-8.0, -4.0, -2.0, 2.0, 4.0, 8.0]:
            cell_key = (cond, k)
            if cell_key in done:
                print(f"[sweep] {cond} k={k:+.1f} — already done, skipping")
                continue
            vec = (k * M * direction).astype(np.float32)
            print(f"\n[sweep] {cond} k={k:+.1f}...")
            res = eval_condition(rows, vec, cond, k)
            rate_dict[k] = res["coop_rate"]
            row = {
                "condition": cond,
                "k": k,
                "coop_rate": res["coop_rate"],
                "unclear_rate": res["unclear_rate"],
                "n_coop": res["n_coop"],
                "n_defect": res["n_defect"],
                "n_unclear": res["n_unclear"],
            }
            sweep_results.append(row)
            _append_partial(row)
            print(f"  coop_rate={res['coop_rate']:.3f}  unclear_rate={res['unclear_rate']:.3f}")

    # Sort for readability
    recog_k = sorted(recog_rates.keys())
    random_k = sorted(random_rates.keys())

    # Spearman correlation: k vs coop_rate for each condition
    from scipy.stats import spearmanr
    recog_k_arr = np.array(recog_k)
    recog_r_arr = np.array([recog_rates[k] for k in recog_k])
    rand_k_arr = np.array(random_k)
    rand_r_arr = np.array([random_rates[k] for k in random_k])

    spearman_recog, pval_recog = spearmanr(recog_k_arr, recog_r_arr)
    spearman_random, pval_random = spearmanr(rand_k_arr, rand_r_arr)

    delta_recog = recog_rates[max(recog_k)] - recog_rates[min(recog_k)]
    delta_random = random_rates[max(random_k)] - random_rates[min(random_k)]

    # Mediation detected if recog shows a meaningful monotone shift AND random stays flat
    mediation_detected = (
        abs(delta_recog) > 0.15
        and abs(spearman_recog) > abs(spearman_random) + 0.2
        and abs(delta_random) < 0.15
    )

    verdict = {
        "M": M,
        "layer": LAYER,
        "rec_features": REC_FEATURES,
        "spearman_recog": {"rho": spearman_recog, "pval": pval_recog},
        "spearman_random": {"rho": spearman_random, "pval": pval_random},
        "delta_coop_rate_recog": delta_recog,
        "delta_coop_rate_random": delta_random,
        "mediation_detected": mediation_detected,
        "notes": (
            "mediation_detected=True means the recog direction produces a meaningful "
            "dose-response (|Δcoop_rate| >0.15, Spearman dominant) while the matched "
            "random control stays flat. False means recognition feature does NOT "
            "causally drive cooperation behavior at this α scale."
        ),
        "recog_rates": {str(k): recog_rates[k] for k in recog_k},
        "random_rates": {str(k): random_rates[k] for k in random_k},
    }

    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    (CAUSAL_DIR / "sweep.json").write_text(json.dumps({"results": sweep_results}, indent=2))
    (CAUSAL_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print()
    print("=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Spearman(k, coop_rate) recog:  ρ={spearman_recog:+.3f}  p={pval_recog:.3f}")
    print(f"  Spearman(k, coop_rate) random: ρ={spearman_random:+.3f}  p={pval_random:.3f}")
    print(f"  Δcoop_rate (k+8→k-8) recog:  {delta_recog:+.3f}")
    print(f"  Δcoop_rate (k+8→k-8) random: {delta_random:+.3f}")
    print(f"  mediation_detected: {mediation_detected}")
    print(f"\n  Recog rates by k:  {recog_rates}")
    print(f"  Random rates by k: {random_rates}")
    print()
    print(f"[sweep] sweep.json  -> {CAUSAL_DIR / 'sweep.json'}")
    print(f"[sweep] verdict.json -> {CAUSAL_DIR / 'verdict.json'}")
    print(f"[sweep] partial log  -> {PARTIAL_PATH}  (left in place for auditing)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Causal mediation test: recognition feature -> cooperation behavior"
    )
    parser.add_argument(
        "--stage", choices=["calibrate", "sweep"], default="calibrate",
        help="calibrate: smoke run on 12 scenarios; sweep: full dose-response on 50 scenarios",
    )
    args = parser.parse_args()

    paths.ensure_run_dirs()

    print("[causal] loading SAE for layer 24...")
    sae = QwenScopeSAE.load(LAYER)
    rec_dir = build_rec_dir(sae).numpy().astype(np.float32)
    rand_dir = build_rand_dir(sae).numpy().astype(np.float32)
    print(f"[causal] rec_dir norm={np.linalg.norm(rec_dir):.4f}  "
          f"rand_dir norm={np.linalg.norm(rand_dir):.4f}")

    print("[causal] computing M (natural recognition signal magnitude)...")
    M = compute_M(torch.tensor(rec_dir))
    print(f"[causal] M = {M:.4f}")

    if args.stage == "calibrate":
        run_calibrate(M, rec_dir)
    else:
        run_sweep(M, rec_dir, rand_dir)


if __name__ == "__main__":
    main()
