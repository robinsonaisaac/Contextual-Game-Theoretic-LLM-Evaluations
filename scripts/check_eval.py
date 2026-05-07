"""Check status of a spawned evaluate() call and pull progress from volume."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-id", default=None,
                    help="If omitted, read from local_data/last_eval_call_id.txt")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--app-name", default="safety")
    ap.add_argument("--volume-name", default="safety")
    ap.add_argument("--peek-only", action="store_true",
                    help="Don't block; just check if done.")
    args = ap.parse_args()

    call_id = args.call_id
    if call_id is None:
        call_id = Path("local_data/last_eval_call_id.txt").read_text().strip()

    fc = modal.FunctionCall.from_id(call_id)
    status = "running"
    result = None
    try:
        # Non-blocking peek
        result = fc.get(timeout=0)
        status = "done"
    except modal.exception.OutputExpiredError:
        status = "expired"
    except TimeoutError:
        status = "running"
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"

    print(f"call_id: {call_id}")
    print(f"status:  {status}")
    if result is not None:
        print("result:")
        print(json.dumps(result, indent=2, default=str))

    # Pull progress JSONL from volume.
    progress_remote = f"runs/{args.run_id}/eval_progress.jsonl"
    progress_local = Path(f"local_data/{args.run_id}_eval_progress.jsonl")
    progress_local.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(
        ["python3", "-m", "modal", "volume", "get", "--force",
         args.volume_name, progress_remote, str(progress_local)],
        stderr=subprocess.DEVNULL,
    )
    if rc == 0 and progress_local.exists():
        lines = progress_local.read_text().splitlines()
        cells = {}
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (rec["tag"], rec["layer"], rec["position"], rec["alpha"])
            cells.setdefault(key, []).append(rec)
        print(f"\nprogress: {len(lines)} story-cell rows across {len(cells)} cells")
        for key, recs in sorted(cells.items()):
            tag, layer, pos, alpha = key
            n = len(recs)
            n_coop = sum(1 for r in recs if r.get("cooperated"))
            print(f"  [{tag}] layer={layer:2d} pos={pos:11s} alpha={alpha:+.1f}: "
                  f"{n_coop}/{n} cooperated")
    else:
        print("\n(no progress JSONL on volume yet)")


if __name__ == "__main__":
    main()
