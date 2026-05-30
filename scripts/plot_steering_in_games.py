#!/usr/bin/env python3
"""Paper figure: transfer of the cooperation vector to multi-agent gameplay.

Two panels (One Night Werewolf | Secret Hitler). Each panel plots the Sonnet
judge's cooperation_index and aggression_index (0-100, 95% CI bars) as a
function of the steering coefficient alpha for the COOPERATION vector, with the
unsteered baseline (alpha=0) shared. The figure shows the same monotonic
cooperation-up / aggression-down dose-response replicating across both games.

Usage:
    python3 scripts/plot_steering_in_games.py \
        --onw data/runs/game_steering_v1/results/aggregate.json \
        --sh  data/runs/sh_steering_v1/results/aggregate.json \
        --out ResultsFigures/steering_in_games.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# cooperation-vector conditions in alpha order; alpha=0 is the shared baseline.
COND_ALPHA = [("coop_a-4", -4.0), ("baseline", 0.0), ("coop_a+4", 4.0)]


def series(agg, metric):
    xs, ys, es = [], [], []
    for label, a in COND_ALPHA:
        if label not in agg:
            continue
        cell = agg[label].get(metric, {})
        if cell.get("mean") is None:
            continue
        xs.append(a)
        ys.append(cell["mean"])
        es.append(cell.get("ci95") or 0.0)
    return xs, ys, es


def panel(ax, agg, title):
    for metric, color, marker, lab in [
        ("cooperation_index", "#2166ac", "o", "cooperation"),
        ("aggression_index", "#b2182b", "s", "aggression"),
    ]:
        xs, ys, es = series(agg, metric)
        ax.errorbar(xs, ys, yerr=es, marker=marker, color=color, capsize=3,
                    lw=2, ms=7, label=lab)
    ax.axvline(0.0, color="0.7", lw=1, ls=":")
    ax.set_xticks([-4, 0, 4])
    ax.set_xlabel(r"cooperation-vector coefficient $\alpha$")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onw", required=True)
    ap.add_argument("--sh", required=True)
    ap.add_argument("--out", default="ResultsFigures/steering_in_games.pdf")
    args = ap.parse_args()

    onw = json.loads(Path(args.onw).read_text())
    sh = json.loads(Path(args.sh).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    panel(axes[0], onw, "One Night Werewolf")
    panel(axes[1], sh, "Secret Hitler")
    axes[0].set_ylabel("judge index (0--100)")
    axes[1].legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
