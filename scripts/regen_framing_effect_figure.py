"""Regenerate fig_cooperation_by_contrast_dim.png as a forest plot.

Shows A-rate difference (lv1 − lv0) per (model, contrast dimension) with 95% CIs.
This replaces the illegible grouped bar chart with a cleaner dot-and-whisker plot.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats

from game_theory_llm.analysis.loader import load_run

RUN_DIR = "data/runs/2026-05-05-sharp"
SAVE_PATH = "-NEURIPS-2026-Framing-The-Game/ResultsFigures/fig_cooperation_by_contrast_dim.png"

MODELS = [
    "gpt-5.4",
    "claude-opus-4.7",
    "claude-sonnet-4.6",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "deepseek-v4-pro",
    "qwen-3.6-plus",
]

MODEL_LABELS = {
    "gpt-5.4": "GPT-5.4",
    "claude-opus-4.7": "Opus 4.7",
    "claude-sonnet-4.6": "Sonnet 4.6",
    "gemini-3.1-pro": "Gemini Pro",
    "gemini-3-flash": "Gemini Flash",
    "deepseek-v4-pro": "DeepSeek",
    "qwen-3.6-plus": "Qwen 3.6+",
}

DIMS = ["gender", "realism", "era", "contrast_domain", "observability"]
DIM_LABELS = {
    "gender": "Gender\n(male − female)",
    "realism": "Realism\n(realistic − fantasy)",
    "era": "Era\n(modern − ancient)",
    "contrast_domain": "Domain\n(political − business)",
    "observability": "Observability\n(public − private)",
}

# Colors for models
COLORS = [
    "#e41a1c",  # GPT
    "#377eb8",  # Opus
    "#4daf4a",  # Sonnet
    "#984ea3",  # Gemini Pro
    "#ff7f00",  # Gemini Flash
    "#a65628",  # DeepSeek
    "#f781bf",  # Qwen
]


def wilson_ci(k, n, z=1.96):
    """Wilson score CI for a proportion. Returns (lower, upper)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - half), min(1, center + half))


def compute_diff_ci(df, model, dim):
    """Compute A-rate difference (lv1 - lv0) with 95% CI via normal approx of difference."""
    col = f"decision_{model}"
    if col not in df.columns:
        return None, None, None

    sub = df[df["contrast_dim"] == dim].dropna(subset=[col, dim])
    levels = sorted(sub[dim].unique())
    if len(levels) < 2:
        return None, None, None

    lv0, lv1 = levels[0], levels[1]
    mask0 = sub[dim] == lv0
    mask1 = sub[dim] == lv1

    n0 = mask0.sum()
    n1 = mask1.sum()
    if n0 == 0 or n1 == 0:
        return None, None, None

    k0 = (sub.loc[mask0, col] == "A").sum()
    k1 = (sub.loc[mask1, col] == "A").sum()
    p0 = k0 / n0
    p1 = k1 / n1
    diff = p1 - p0

    # SE of difference via normal approx
    se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se
    return diff, ci_lo, ci_hi


def main():
    print("Loading data...")
    df = load_run(RUN_DIR, include_swapped=False)
    print(f"Loaded {len(df)} rows")

    n_dims = len(DIMS)
    n_models = len(MODELS)

    # Row labels on the left panel identify the models, so no legend is needed;
    # the LaTeX caption supplies the title. Both are dropped to keep the float short.
    fig, axes = plt.subplots(1, n_dims, figsize=(3.2 * n_dims, 3.1), sharey=False)

    y_positions = np.arange(n_models)

    for ax_idx, (dim, ax) in enumerate(zip(DIMS, axes)):
        diffs, los, his = [], [], []
        for model in MODELS:
            d, lo, hi = compute_diff_ci(df, model, dim)
            diffs.append(d)
            los.append(lo)
            his.append(hi)

        for i, (model, d, lo, hi, color) in enumerate(
            zip(MODELS, diffs, los, his, COLORS)
        ):
            if d is None:
                continue
            err_lo = d - lo
            err_hi = hi - d

            # Mark significant result with a filled marker
            is_sig = (model == "gpt-5.4") and (dim == "realism")
            marker = "D" if is_sig else "o"
            ms = 7 if is_sig else 5
            zorder = 5 if is_sig else 3

            ax.errorbar(
                d, i,
                xerr=[[err_lo], [err_hi]],
                fmt=marker,
                color=color,
                markersize=ms,
                capsize=3,
                linewidth=1.2,
                zorder=zorder,
            )
            # The diamond marker alone flags the significant cell (see caption);
            # no extra glyph, which would go unexplained.

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_title(DIM_LABELS[dim], fontsize=8.5, pad=4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            fontsize=8,
        )
        ax.set_ylim(-0.7, n_models - 0.3)

        # x-axis range: symmetric around 0, at least ±0.25
        max_abs = max(
            (abs(d) + max(abs(d - lo), abs(hi - d)) for d, lo, hi in zip(diffs, los, his) if d is not None),
            default=0.25,
        )
        xlim = max(0.25, max_abs * 1.3)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Δ A-rate", fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)

        if ax_idx > 0:
            ax.set_yticklabels([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    fig.savefig(SAVE_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved to {SAVE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
