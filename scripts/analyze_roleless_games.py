#!/usr/bin/env python3
"""Objective, judge-free analysis of the role-free steering batteries.

Everything here is computed from numbers the players chose or the engine
scored — no LLM judge is involved at any point. That is the whole reason these
two games exist: the gameplay result they replace was an artifact of what a
judge was shown, so the replacement instrument has no judge in it.

Dependent variables
-------------------
public_goods
    mean_contribution_rate   primary: cooperative intent, 0..1
    group_efficiency         realised / socially optimal earnings. NOTE the
                             floor is 1/multiplier (0.5 at m=2), not 0 —
                             universal defection still banks n*E per round.
    free_ride_rate           share of choices below 20% of endowment
    decay_slope              OLS slope of contribution rate on round index;
                             the classic finding is a negative slope, and
                             whether steering changes the SLOPE rather than the
                             level is a separate question from the mean.
hanabi
    score                    primary: 0..25 under the configured bomb rule
    fireworks_score          raw total, unaffected by the strict-bomb zeroing
    bomb_rate                share of matches ending on three fuses

Reads the launcher's manifest (each row carries a ``metrics`` dict written by
the worker) and falls back to re-deriving from the match logs for any row that
predates that field.

Usage:
    python3 scripts/analyze_roleless_games.py data/runs/pg_mixed_v1
    python3 scripts/analyze_roleless_games.py data/runs/hanabi_mixed_v1 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats as _sps
except Exception:                                    # pragma: no cover
    _sps = None

PRIMARY = {
    "public_goods": "mean_contribution_rate",
    "hanabi": "score",
}


# ------------------------------------------------------------------ helpers
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ci95(xs):
    """Normal-approximation 95% CI half-width of the mean."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return 1.96 * sd / math.sqrt(n)


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U p-value, or None without scipy."""
    if _sps is None or len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(_sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def ols_slope(ys):
    """Slope of y on its index — the contribution decay curve."""
    n = len(ys)
    if n < 2:
        return float("nan")
    xs = list(range(n))
    mx, my = mean(xs), mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def terminal_of(log_path: Path):
    term = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "terminal":
            term = ev
    return term


def derive_metrics(game: str, row: dict) -> dict:
    """Prefer the worker's metrics; otherwise recover them from the log."""
    m = row.get("metrics") or {}
    if m:
        return m
    log_path = Path(row.get("log", ""))
    if not log_path.exists():
        return {}
    term = terminal_of(log_path)
    if not term:
        return {}
    snap = term.get("state")
    if isinstance(snap, str):        # _safe_dump fell back to str(state)
        return {}
    if not isinstance(snap, dict):
        return {}
    if game == "public_goods":
        return {k: snap.get(k) for k in
                ("mean_contribution_rate", "group_efficiency",
                 "free_ride_rate", "round_contribution_rates")}
    return {"score": snap.get("score"),
            "fireworks_score": snap.get("fireworks_score"),
            "end_reason": snap.get("end_reason")}


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None, help="write the full table as JSON")
    args = ap.parse_args()

    run = Path(args.run_dir)
    rows = json.loads((run / "manifest.json").read_text())
    jobs = json.loads((run / "jobs.json").read_text()) if (run / "jobs.json").exists() else {}
    game = jobs.get("game") or ("public_goods" if any(
        "contribution" in json.dumps(r.get("metrics", {})) for r in rows) else "hanabi")

    finished = [r for r in rows if r.get("winner")]
    print(f"run: {run}")
    print(f"game: {game}   rows: {len(rows)}   completed: {len(finished)} "
          f"({len(finished) / max(len(rows), 1):.1%})")
    if not finished:
        raise SystemExit("no completed matches yet")

    # cell -> list of metric dicts
    cells = defaultdict(list)
    for r in finished:
        cells[(r["label"], r["v"], r["alpha"])].append(derive_metrics(game, r))

    primary = PRIMARY[game]
    baseline_key = next((k for k in cells if k[0] == "baseline"), None)
    base_vals = [m.get(primary) for m in cells.get(baseline_key, [])
                 if m.get(primary) is not None] if baseline_key else []

    print(f"\nprimary DV: {primary}"
          + (f"   baseline mean {mean(base_vals):.4f} (n={len(base_vals)})"
             if base_vals else ""))
    header = f"{'cell':>12} {'k':>2} {'alpha':>6} {'n':>4} {'mean':>9} {'+/-95%':>8} {'p vs base':>10}"
    print("\n" + header)
    print("-" * len(header))

    table = []
    for (label, k, alpha) in sorted(cells, key=lambda x: (x[1], x[2])):
        ms = cells[(label, k, alpha)]
        vals = [m.get(primary) for m in ms if m.get(primary) is not None]
        if not vals:
            continue
        p = mannwhitney(base_vals, vals) if base_vals and label != "baseline" else None
        table.append({"cell": label, "k": k, "alpha": alpha, "n": len(vals),
                      "mean": mean(vals), "ci95": ci95(vals), "p_vs_baseline": p})
        print(f"{label:>12} {k:>2} {alpha:>6.1f} {len(vals):>4} {mean(vals):>9.4f} "
              f"{ci95(vals):>8.4f} " + (f"{p:>10.4g}" if p is not None else f"{'-':>10}"))

    # ---- secondary measures -------------------------------------------
    print("\nsecondary measures by cell:")
    for (label, k, alpha) in sorted(cells, key=lambda x: (x[1], x[2])):
        ms = cells[(label, k, alpha)]
        if game == "public_goods":
            eff = [m["group_efficiency"] for m in ms if m.get("group_efficiency") is not None]
            fr = [m["free_ride_rate"] for m in ms if m.get("free_ride_rate") is not None]
            slopes = [ols_slope(m["round_contribution_rates"]) for m in ms
                      if m.get("round_contribution_rates")]
            print(f"  {label:>12}  efficiency {mean(eff):.4f}   free-ride {mean(fr):.4f}   "
                  f"decay slope {mean(slopes):+.4f}")
        else:
            raw = [m["fireworks_score"] for m in ms if m.get("fireworks_score") is not None]
            bombs = [1.0 if m.get("end_reason") == "bombed" else 0.0 for m in ms
                     if m.get("end_reason")]
            print(f"  {label:>12}  raw fireworks {mean(raw):.3f}   "
                  f"bomb rate {mean(bombs):.3f}")

    if game == "public_goods":
        print("\nnote: group_efficiency floor is 1/multiplier (0.5 at m=2), not 0 — "
              "universal defection still banks n*E per round.")
    else:
        # Under the strict convention a bombed match scores 0, so a population
        # that bombs nearly everything collapses the primary DV to a constant
        # and no contrast can be detected however large the true effect is.
        # Say so loudly rather than reporting a row of zeroes as a null.
        allm = [m for ms in cells.values() for m in ms]
        bombed = [1.0 if m.get("end_reason") == "bombed" else 0.0
                  for m in allm if m.get("end_reason")]
        if bombed and mean(bombed) > 0.8:
            print(f"\n*** WARNING: {mean(bombed):.0%} of matches ended on three "
                  "fuses. Under strict_bombs the primary DV is ~0 everywhere and "
                  "this analysis cannot detect an effect of any size. Use "
                  "fireworks_score (reported above) as the primary measure, or "
                  "re-run with strict_bombs=False. ***")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"run": str(run), "game": game, "primary": primary,
             "completed": len(finished), "cells": table}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
