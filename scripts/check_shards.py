"""Check status of all shards spawned by spawn_eval_parallel.py."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-ids", default="local_data/last_eval_shard_call_ids.json")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--volume-name", default="safety")
    args = ap.parse_args()

    blob = json.loads(Path(args.call_ids).read_text())
    run_id = args.run_id or blob["run_id"]
    shards = blob["shards"]

    print(f"run_id: {run_id}")
    print(f"shards: {len(shards)}")
    print()

    n_done = 0
    n_running = 0
    n_other = 0
    rows = []
    for sh in shards:
        fc = modal.FunctionCall.from_id(sh["call_id"])
        try:
            result = fc.get(timeout=0)
            status = "done"
            n_done += 1
        except TimeoutError:
            status = "running"
            result = None
            n_running += 1
        except Exception as e:
            status = f"err:{type(e).__name__}"
            result = None
            n_other += 1
        coop = ""
        if result is not None and isinstance(result, dict):
            coop = f"  coop={result.get('n_cooperated')}/{result.get('n_stories')}"
        # Single-vector vs multi-vector shards have different keys
        if "label" in sh:
            label = sh["label"]
        else:
            label = f"L{sh['layer']:2d} {sh['position']:11s}"
        rows.append((label, sh["alpha"], status, coop))

    rows.sort()
    for label, alpha, status, coop in rows:
        print(f"  {label} α={alpha:+.1f}: {status}{coop}")

    print()
    print(f"summary: done={n_done}/{len(shards)} running={n_running} other={n_other}")

    # Also pull progress JSONL for a fine-grained view.
    progress_remote = f"runs/{run_id}/eval_progress.jsonl"
    progress_local = Path(f"local_data/{run_id}_eval_progress.jsonl")
    progress_local.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(
        ["python3", "-m", "modal", "volume", "get", "--force",
         args.volume_name, progress_remote, str(progress_local)],
        stderr=subprocess.DEVNULL,
    )
    if rc == 0 and progress_local.exists():
        n_lines = sum(1 for _ in progress_local.open())
        print(f"progress lines on volume: {n_lines}")


if __name__ == "__main__":
    main()
