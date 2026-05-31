"""Score + aggregate the capability-regression ablation (GSM8k, HumanEval).

Pulls the generation-trace shards written by run_capability_evals.py from the
Modal volume, joins them with the local corpus (gold answers / unit tests) by
story_id, scores LOCALLY (numeric match for GSM8k; sandboxed subprocess code
execution for HumanEval), and aggregates accuracy / pass@1 per (vector, alpha)
with Wilson CIs and Fisher-exact contrasts vs the shared alpha=0 baseline.

Usage:
    python3 scripts/analyze_capability_evals.py            # pull + score all
    python3 scripts/analyze_capability_evals.py --no-pull  # reuse local shards
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

from game_theory_llm.capability_scoring import gsm8k_correct, humaneval_passes

VECTORS = {"cooperation": "pd_full_v1", "trust": "pd_E4B_trust_v1"}
BENCHMARKS = ["gsm8k", "humaneval"]
VOLUME = "safety"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pull_shards(run_id, benchmark):
    local = Path(f"local_data/capability/{run_id}_shards_{benchmark}")
    local.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python3", "-m", "modal", "volume", "get", "--force", VOLUME,
        f"runs/{run_id}/shards_{benchmark}/", str(local)])
    return local


def load_traces(local_dir):
    pqs = sorted(Path(local_dir).rglob("*.parquet"))
    if not pqs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    return df[["alpha", "story_id", "trace"]]


def score_rows(df, benchmark, corpus):
    """Return df with a 'correct' bool column, scoring each trace locally."""
    if benchmark == "gsm8k":
        def sc(r):
            meta = corpus.get(r["story_id"])
            return bool(meta) and gsm8k_correct(r["trace"], meta["gsm8k_gold"])
        df = df.copy()
        df["correct"] = df.apply(sc, axis=1)
        return df
    # humaneval: execute in parallel (subprocesses with timeout)
    rows = df.to_dict("records")

    def sc(r):
        meta = corpus.get(r["story_id"])
        if not meta:
            return False
        return humaneval_passes(r["trace"], meta["he_prompt"],
                                meta["he_test"], meta["he_entry_point"])
    with ThreadPoolExecutor(max_workers=8) as ex:
        correct = list(ex.map(sc, rows))
    df = df.copy()
    df["correct"] = correct
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--out", default="data/runs/capability/results")
    args = ap.parse_args()

    corpora = {b: {json.loads(l)["story_id"]: json.loads(l)
                   for l in Path(f"data/runs/capability/{b}_eval.jsonl").read_text().splitlines() if l.strip()}
               for b in BENCHMARKS}

    agg = {}        # benchmark -> list of rows
    per_item = []
    for benchmark in BENCHMARKS:
        scored_by_vec = {}
        for vector, run_id in VECTORS.items():
            local = (Path(f"local_data/capability/{run_id}_shards_{benchmark}")
                     if args.no_pull else pull_shards(run_id, benchmark))
            df = load_traces(local)
            if df.empty:
                print(f"[warn] no shards for {vector}/{benchmark}")
                continue
            df = score_rows(df, benchmark, corpora[benchmark])
            df["vector"] = vector
            df["benchmark"] = benchmark
            scored_by_vec[vector] = df
            per_item.append(df)

        # baseline = cooperation alpha==0 (shared)
        base = None
        if "cooperation" in scored_by_vec:
            b = scored_by_vec["cooperation"]
            base = b[b["alpha"] == 0.0]
        rows = []
        for vector, df in scored_by_vec.items():
            for alpha, g in df.groupby("alpha"):
                if vector == "trust" and alpha == 0.0:
                    continue
                k, n = int(g["correct"].sum()), len(g)
                lo, hi = wilson_ci(k, n)
                p = None
                if base is not None and not (vector == "cooperation" and alpha == 0.0):
                    bk, bn = int(base["correct"].sum()), len(base)
                    _, p = fisher_exact([[k, n - k], [bk, bn - bk]])
                rows.append({"vector": vector, "benchmark": benchmark,
                             "alpha": float(alpha), "n": n, "k": k,
                             "acc": k / n if n else None,
                             "ci_lo": lo, "ci_hi": hi, "fisher_p_vs_baseline": p})
        agg[benchmark] = sorted(rows, key=lambda r: (r["vector"], r["alpha"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    if per_item:
        pd.concat(per_item, ignore_index=True)[
            ["benchmark", "vector", "alpha", "story_id", "correct"]
        ].to_parquet(out / "per_item.parquet", index=False)

    for benchmark in BENCHMARKS:
        print(f"\n=== {benchmark} ===")
        print(f"{'vector':12s} {'alpha':>6} {'n':>4} {'acc':>7} {'95% CI':>16} {'Fisher p':>9}")
        for r in agg.get(benchmark, []):
            ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
            p = f"{r['fisher_p_vs_baseline']:.3f}" if r["fisher_p_vs_baseline"] is not None else "  base"
            acc = f"{r['acc']:.3f}" if r["acc"] is not None else "   -"
            print(f"{r['vector']:12s} {r['alpha']:>+6g} {r['n']:>4} {acc:>7} {ci:>16} {p:>9}")
    print(f"\nwrote {out}/aggregate.json + per_item.parquet")


if __name__ == "__main__":
    main()
