#!/usr/bin/env python3
"""Re-judge the UNIFORM-steering runs with the focused per-seat instrument.

`game_steering_v2` and `sh_steering_v2` are the runs behind the paper. Their
manifests predate the mixed-population schema, so they carry no `roles`,
`treat_seats`, `w` or `v` fields. This adapter reconstructs them:

  roles        <- the setup record's god_view (initial_roles for ONW, roles for SH)
  treat_seats  <- every seat, because these runs steered all seats uniformly
  w / v        <- counts implied by those roles

It then reuses `analyze_mixed_population` unchanged, so the uniform and
mixed-population numbers land on exactly the same instrument: one focused judge
call per seat, whole transcript, judge blind to steering condition.

The published numbers came from a DIFFERENT instrument -- one whole-table call
rating all seats at once on three collinear scales, over a talk-only transcript
truncated at 9,000 characters. Output here is therefore not a correction of
those numbers but a second measurement of the same matches.

Usage:
    python3 scripts/rejudge_uniform_runs.py --run data/runs/game_steering_v2 \
        --game one_night_werewolf --out data/runs/game_steering_v2/per_seat_focused.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib.machinery import SourceFileLoader

_A = SourceFileLoader(
    "analyze_mixed_population",
    str(Path(__file__).resolve().parent / "analyze_mixed_population.py"),
).load_module()

HIDDEN = {"one_night_werewolf": {"Werewolf"},
          "secret_hitler": {"Fascist", "Hitler"}}


def roles_from_log(log_path: Path):
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "setup":
            gv = rec.get("god_view", {}) or {}
            return list(gv.get("initial_roles") or gv.get("roles") or [])
    return []


def adapt(manifest, game):
    """Rewrite uniform-run manifest entries into the mixed-population schema."""
    hidden = HIDDEN[game]
    out = []
    for m in manifest:
        log = Path(m["log"])
        if not log.exists():
            continue
        roles = roles_from_log(log)
        if not roles:
            continue
        alpha = float(m.get("alpha") or 0.0)
        # uniform runs steer every seat; at alpha=0 nothing is steered
        treat = list(range(len(roles))) if alpha != 0.0 else []
        w = sum(1 for i in treat if roles[i] in hidden)
        out.append({**m, "roles": roles, "treat_seats": treat,
                    "w": w, "v": len(treat) - w})
    return out


async def main_async(args):
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    entries = adapt(manifest, args.game)
    print(f"[rejudge] {run.name}: {len(entries)}/{len(manifest)} matches adapted "
          f"({args.game})")

    from game_theory_llm.client import LLMClient
    client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[
        _A.process(e, client, sem, False, args.game) for e in entries])
    rows = [r for grp in results if grp for r in grp]
    out = Path(args.out or (run / "per_seat_focused.json"))
    out.write_text(json.dumps(rows, indent=2))
    scored = sum(1 for r in rows if r.get("coop_judged") is not None)
    print(f"[rejudge] wrote {len(rows)} seat-rows ({scored} scored) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--game", required=True, choices=sorted(HIDDEN))
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
