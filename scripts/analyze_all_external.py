"""Comprehensive analyzer for all external moral benchmark shards
(MoralBench, MoralChoice, ETHICS-Util, ETHICS-Deontology) on E4B and
26B-A4B. Produces:

  - Wilson 95% CIs on per-alpha alignment rates
  - Fisher exact 2×2 p-value for Δ(α=+3, α=−3)
  - Newcombe-Wilson 95% CI on the difference of proportions
  - A unified summary table across all (model, benchmark) cells.

Usage:
    python3 scripts/analyze_all_external.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BENCHMARKS = [
    # (label, run_id, subdir, eval_path, model)
    ("MoralBench (E4B)",
     "pd_full_v1", "shards_moralbench",
     "data/runs/moralbench/eval.jsonl", "E4B"),
    ("MoralChoice low-ambig (E4B)",
     "pd_full_v1", "shards_moralchoice",
     "data/runs/moralchoice/eval.jsonl", "E4B"),
    ("MoralChoice high-ambig (E4B)",
     "pd_full_v1", "shards_moralchoice",
     "data/runs/moralchoice/eval.jsonl", "E4B"),
    ("ETHICS-Util (E4B)",
     "pd_full_v1", "shards_ethics_util",
     "data/runs/ethics_util/eval.jsonl", "E4B"),
    ("ETHICS-Deontology (E4B)",
     "pd_full_v1", "shards_ethics_deontology",
     "data/runs/ethics_deontology/eval.jsonl", "E4B"),
    ("MoralChoice high-ambig (26B-A4B)",
     "pd_26B_A4B_v1", "shards_moralchoice_high",
     "data/runs/moralchoice/eval_high_subset100.jsonl", "26B-A4B"),
    ("ETHICS-Util (26B-A4B)",
     "pd_26B_A4B_v1", "shards_ethics_util",
     "data/runs/ethics_util/eval_subset100.jsonl", "26B-A4B"),
    ("ETHICS-Deontology (26B-A4B)",
     "pd_26B_A4B_v1", "shards_ethics_deontology",
     "data/runs/ethics_deontology/eval_subset100.jsonl", "26B-A4B"),
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_wilson_diff_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe's hybrid score CI for the difference of two proportions
    (p2 − p1), based on Wilson intervals for each. Conservative but
    well-behaved at extreme proportions."""
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    delta = p2 - p1
    lo = delta - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = delta + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def _vol_get(remote: str, local: str) -> int:
    return subprocess.call(
        ["python3", "-m", "modal", "volume", "get", "--force",
         "safety", remote, local],
        stderr=subprocess.DEVNULL,
    )


def _load_meta(eval_path: Path) -> dict[str, dict]:
    meta = {}
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta[obj["story_id"]] = {
            "coop_choice": obj.get("coop_choice"),
            "tag": obj.get("framing", obj.get("game_type", "")),
            "score_A": obj.get("human_score_A"),
            "score_B": obj.get("human_score_B"),
        }
    return meta


def analyze_one(label: str, run_id: str, subdir: str,
                eval_path: str, model: str) -> list[dict]:
    eval_path_p = Path(eval_path)
    if not eval_path_p.exists():
        return []
    meta = _load_meta(eval_path_p)

    dl = Path(f"local_data/{run_id}_{subdir}_dl")
    dl.mkdir(parents=True, exist_ok=True)
    _vol_get(f"runs/{run_id}/{subdir}/", str(dl))
    parquets = sorted(dl.rglob("*.parquet"))
    if not parquets:
        print(f"[{label}] no parquets")
        return []

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df["coop_choice"] = df["story_id"].map(lambda s: meta.get(s, {}).get("coop_choice"))
    df["tag"] = df["story_id"].map(lambda s: meta.get(s, {}).get("tag", ""))

    # MoralBench has human_score_A/B and no clean coop_choice; use higher-score-wins
    if "MoralBench" in label:
        df["score_A"] = df["story_id"].map(lambda s: meta.get(s, {}).get("score_A"))
        df["score_B"] = df["story_id"].map(lambda s: meta.get(s, {}).get("score_B"))
        df["coop_choice"] = df.apply(
            lambda r: ("A" if (r["score_A"] is not None and r["score_B"] is not None
                               and r["score_A"] >= r["score_B"]) else "B"),
            axis=1,
        )
    # MoralChoice low/high split
    if "MoralChoice low-ambig" in label:
        df = df[df["tag"] == "low"]
    elif "MoralChoice high-ambig" in label and "26B" not in label:
        df = df[df["tag"] == "high"]

    parsed = df[df["decision"].isin(["A", "B"])].copy()
    parsed["correct"] = (parsed["decision"] == parsed["coop_choice"]).astype(int)

    rows = []
    for alpha, sub in parsed.groupby("alpha"):
        k = int(sub["correct"].sum())
        n = int(len(sub))
        lo, hi = wilson_ci(k, n)
        rows.append({
            "benchmark": label, "model": model, "alpha": alpha,
            "k": k, "n": n, "rate": k / n if n else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
        })
    return rows


def summarize_diff(rows: list[dict]) -> pd.DataFrame:
    """For each benchmark, compute Δ(α=+3, α=−3) with Newcombe CI and Fisher p."""
    out = []
    df = pd.DataFrame(rows)
    for (bm, model), sub in df.groupby(["benchmark", "model"]):
        row_m3 = sub[sub["alpha"] == -3.0]
        row_p3 = sub[sub["alpha"] == 3.0]
        if row_m3.empty or row_p3.empty:
            continue
        k1, n1 = int(row_m3.iloc[0]["k"]), int(row_m3.iloc[0]["n"])
        k2, n2 = int(row_p3.iloc[0]["k"]), int(row_p3.iloc[0]["n"])
        lo, hi = newcombe_wilson_diff_ci(k1, n1, k2, n2)
        # Fisher exact
        table = [[k2, n2 - k2], [k1, n1 - k1]]
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = float("nan")
        out.append({
            "benchmark": bm, "model": model,
            "rate_-3": k1 / n1, "rate_+3": k2 / n2,
            "delta": (k2 / n2) - (k1 / n1),
            "delta_ci_lo": lo, "delta_ci_hi": hi,
            "n": n1, "fisher_p": p,
        })
    return pd.DataFrame(out)


def main():
    all_rows = []
    for label, run_id, subdir, eval_path, model in BENCHMARKS:
        rows = analyze_one(label, run_id, subdir, eval_path, model)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).sort_values(["benchmark", "alpha"]).reset_index(drop=True)
    print("\n=== Per-alpha alignment rate with Wilson 95% CIs ===")
    for (bm, model), sub in df.groupby(["benchmark", "model"]):
        print(f"\n  --- {bm} ---")
        for _, r in sub.iterrows():
            print(f"    α={r['alpha']:+.1f}  {r['k']:>3}/{r['n']:>3} = "
                  f"{r['rate']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    summary = summarize_diff(all_rows)
    print("\n\n=== Δ(α=+3 minus α=−3) with Newcombe-Wilson 95% CI and Fisher exact p ===")
    print(summary.round(3).to_string(index=False))

    Path("local_data/external_full_summary.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet("local_data/external_full_summary.parquet")
    summary.to_parquet("local_data/external_delta_summary.parquet")
    print("\n[saved] local_data/external_full_summary.parquet")
    print("[saved] local_data/external_delta_summary.parquet")


if __name__ == "__main__":
    main()
