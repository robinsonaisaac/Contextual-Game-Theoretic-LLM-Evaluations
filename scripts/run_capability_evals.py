"""Run the steering capability-regression ablation: GSM8k (math) + HumanEval
(coding), sweeping BOTH steering vectors (cooperation pd_full_v1, trust
pd_E4B_trust_v1) at alpha in {-6,-3,0,+3,+6} on Gemma 4 E4B-it, L16 mean_trace.

This reuses the deployed SteeringWorker.eval_shard (which persists the full
generation trace to the volume); scoring is done locally afterward by
scripts/analyze_capability_evals.py (numeric match for GSM8k, sandboxed code
execution for HumanEval). No Modal redeploy needed.

alpha=0 (no steering) is identical across vectors, so we run the baseline once
under the cooperation run_id and reuse it for both vectors. Shards land at
/data/runs/{run_id}/shards_{benchmark}/.

Usage:
    python3 scripts/run_capability_evals.py
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

HF_ID = "google/gemma-4-E4B-it"
LAYER = 16
POSITION = "mean_trace"
COOP_RUN_ID = "pd_full_v1"
TRUST_RUN_ID = "pd_E4B_trust_v1"
ALPHAS = (-6.0, -3.0, 0.0, 3.0, 6.0)
BENCHMARKS = ["gsm8k", "humaneval"]


def jobs():
    """(vector, run_id, benchmark, alpha). Baseline (alpha=0) only under coop."""
    for bench in BENCHMARKS:
        for a in ALPHAS:
            yield ("cooperation", COOP_RUN_ID, bench, a)
        for a in ALPHAS:
            if a == 0.0:
                continue                     # share the coop baseline
            yield ("trust", TRUST_RUN_ID, bench, a)


def main():
    Cls = modal.Cls.from_name("safety", "SteeringWorker")
    worker = Cls(model_name=HF_ID)

    corpora = {b: [json.loads(l) for l in
                   Path(f"data/runs/capability/{b}_eval.jsonl").read_text().splitlines() if l.strip()]
               for b in BENCHMARKS}
    for b, rows in corpora.items():
        print(f"[corpus] {b}: {len(rows)} items")

    specs = list(jobs())
    print(f"[exp] spawning {len(specs)} eval shards "
          f"(2 benchmarks x vectors x alphas)")

    def spawn(spec):
        vector, run_id, bench, alpha = spec
        fc = worker.eval_shard.spawn(
            run_id=run_id, layer=LAYER, position=POSITION, alpha=float(alpha),
            stories=corpora[bench], result_subdir=f"shards_{bench}")
        return {"vector": vector, "run_id": run_id, "benchmark": bench,
                "alpha": float(alpha), "call_id": fc.object_id}

    manifest = [spawn(s) for s in specs]
    out = Path("data/runs/capability")
    out.mkdir(parents=True, exist_ok=True)
    (out / "calls.json").write_text(json.dumps(manifest, indent=2))
    print(f"[exp] spawned; collecting (out of order)...")

    def fetch(m):
        try:
            res = modal.FunctionCall.from_id(m["call_id"]).get(timeout=3000)
            return m, res, None
        except Exception as e:
            return m, None, f"{type(e).__name__}: {e}"

    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        for fut in as_completed([ex.submit(fetch, m) for m in manifest]):
            m, res, err = fut.result()
            done += 1
            if err:
                print(f"[exp] ERR {m['vector']}/{m['benchmark']} a={m['alpha']:+g}: {err}")
            else:
                print(f"[exp] {done}/{len(manifest)} {m['vector']}/{m['benchmark']} "
                      f"a={m['alpha']:+g} n={res.get('n_stories')} -> {res.get('shard_path')}")
    print(f"[exp] done: {done}/{len(manifest)} shards. Now run analyze_capability_evals.py")


if __name__ == "__main__":
    main()
