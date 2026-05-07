"""Regenerate ResultsFigures/Historical_Cooperation.{png,pdf}.

Changes from the original figure:
  - GPT-4 dropped (outlier in time; distorts x-axis scale)
  - "RLHF shift" annotation removed
  - Cleaner styling: lighter reference lines, consistent label positions,
    tighter x-axis padding, unified font sizes
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DIR = Path("data/runs/2026-05-05-sharp/evals_historical")
SAVE_PDF = Path("-NEURIPS-2026-Framing-The-Game/ResultsFigures/Historical_Cooperation.pdf")
SAVE_PNG = Path("-NEURIPS-2026-Framing-The-Game/ResultsFigures/Historical_Cooperation.png")

# Models to include, grouped into families (ordered oldest → newest).
# GPT-4 deliberately excluded.
FAMILIES = {
    "GPT": {
        "color": "#2ca02c",   # green
        "marker": "o",
        "models": [
            "gpt-4o",
            "gpt-4.1",
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.4",
        ],
    },
    "Claude Opus": {
        "color": "#ff7f0e",   # orange
        "marker": "s",
        "models": [
            "claude-opus-4",
            "claude-opus-4.1",
            "claude-opus-4.5",
            "claude-opus-4.6",
            "claude-opus-4.7",
        ],
    },
    "Gemini": {
        "color": "#1f77b4",   # blue
        "marker": "^",
        "models": [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-pro",
        ],
    },
    "Grok": {
        "color": "#333333",   # near-black
        "marker": "D",
        "models": [
            "grok-3-beta",
            "grok-3",
            "grok-4",
            "grok-4-fast",
            "grok-4.20",
            "grok-4.3",
        ],
    },
    "Qwen": {
        "color": "#9467bd",   # purple
        "marker": "P",
        "models": [
            "qwen-2.5-72b",
            "qwen-3-235b-a22b",
            "qwen-3-max",
            "qwen-3.5-397b",
            "qwen-3.6-plus",
        ],
    },
}

# Approximate release dates (YYYY-MM-DD).  Used solely for the x-axis.
RELEASE_DATES = {
    # GPT (post-GPT-4)
    "gpt-4o":       "2024-05-13",
    "gpt-4.1":      "2025-04-14",
    "gpt-5":        "2025-02-27",
    "gpt-5.1":      "2025-04-07",
    "gpt-5.2":      "2025-05-05",
    "gpt-5.4":      "2026-03-10",
    # Claude Opus
    "claude-opus-4":   "2025-07-15",
    "claude-opus-4.1": "2025-08-20",
    "claude-opus-4.5": "2025-11-03",
    "claude-opus-4.6": "2026-01-20",
    "claude-opus-4.7": "2026-03-15",
    # Gemini
    "gemini-2.0-flash":  "2025-06-01",
    "gemini-2.5-flash":  "2025-08-05",
    "gemini-2.5-pro":    "2025-09-10",
    "gemini-3-flash":    "2025-12-10",
    "gemini-3.1-pro":    "2026-03-20",
    # Grok
    "grok-3-beta":  "2025-05-05",
    "grok-3":       "2025-06-10",
    "grok-4":       "2025-12-15",
    "grok-4-fast":  "2026-01-15",
    "grok-4.20":    "2026-04-20",
    "grok-4.3":     "2026-06-01",
    # Qwen
    "qwen-2.5-72b":     "2024-08-20",
    "qwen-3-235b-a22b": "2025-09-15",
    "qwen-3-max":       "2025-10-10",
    "qwen-3.5-397b":    "2026-02-10",
    "qwen-3.6-plus":    "2026-06-10",
}

# Short display labels (only first and last in each family get labeled)
DISPLAY_LABELS = {
    "gpt-4o":           "GPT-4o",
    "gpt-4.1":          "GPT-4.1",
    "gpt-5":            "GPT-5",
    "gpt-5.1":          "GPT-5.1",
    "gpt-5.2":          "GPT-5.2",
    "gpt-5.4":          "GPT-5.4",
    "claude-opus-4":    "Opus 4",
    "claude-opus-4.1":  "Opus 4.1",
    "claude-opus-4.5":  "Opus 4.5",
    "claude-opus-4.6":  "Opus 4.6",
    "claude-opus-4.7":  "Opus 4.7",
    "gemini-2.0-flash": "2.0-flash",
    "gemini-2.5-flash": "2.5-flash",
    "gemini-2.5-pro":   "2.5-pro",
    "gemini-3-flash":   "3-flash",
    "gemini-3.1-pro":   "3.1-pro",
    "grok-3-beta":      "Grok 3β",
    "grok-3":           "Grok 3",
    "grok-4":           "Grok 4",
    "grok-4-fast":      "Grok 4-fast",
    "grok-4.20":        "Grok 4.20",
    "grok-4.3":         "Grok 4.3",
    "qwen-2.5-72b":     "Qwen 2.5",
    "qwen-3-235b-a22b": "Qwen 3",
    "qwen-3-max":       "Qwen 3-Max",
    "qwen-3.5-397b":    "Qwen 3.5",
    "qwen-3.6-plus":    "Qwen 3.6",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_decision(response: str):
    """Return 'A' or 'B' from a model response, or None if unparseable."""
    if not isinstance(response, str):
        return None
    m = re.search(r"<decision>\s*([AB])\s*</decision>", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def load_cooperation_rates():
    """Return {model_name: cooperation_rate} from evals_historical/."""
    totals: dict[str, dict] = defaultdict(lambda: {"a": 0, "n": 0})
    for path in RUN_DIR.glob("prisoners_dilemma__*.jsonl"):
        model = path.stem.rsplit("__", 1)[-1]
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = parse_decision(rec.get("response", ""))
                if decision is not None:
                    totals[model]["n"] += 1
                    if decision == "A":
                        totals[model]["a"] += 1
    return {
        model: d["a"] / d["n"]
        for model, d in totals.items()
        if d["n"] > 0
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    rates = load_cooperation_rates()
    print(f"Loaded cooperation rates for {len(rates)} models")

    fig, ax = plt.subplots(figsize=(11, 5))

    # Faint reference lines
    for y, lbl in [(0.5, "50%"), (0.9, "90%")]:
        ax.axhline(y, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(datetime(2024, 4, 1), y + 0.008, lbl,
                fontsize=7.5, color="#aaaaaa", va="bottom")

    legend_handles = []

    for family, cfg in FAMILIES.items():
        color   = cfg["color"]
        marker  = cfg["marker"]
        models  = cfg["models"]

        xs, ys, ms = [], [], []
        for m in models:
            if m not in rates:
                continue
            if m not in RELEASE_DATES:
                continue
            xs.append(datetime.strptime(RELEASE_DATES[m], "%Y-%m-%d"))
            ys.append(rates[m])
            ms.append(m)

        if not xs:
            continue

        # Sort by date
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ms = [ms[i] for i in order]

        line, = ax.plot(xs, ys, color=color, marker=marker,
                        markersize=6, linewidth=1.4, zorder=3, label=family)
        legend_handles.append(line)

        # Label only first and last checkpoint in each family
        label_indices = {0, len(xs) - 1}
        for i, (x, y, m) in enumerate(zip(xs, ys, ms)):
            if i not in label_indices:
                continue
            lbl = DISPLAY_LABELS.get(m, m)
            # Offset direction: first → left, last → right
            if i == 0:
                ha, xoff = "right", -4
            else:
                ha, xoff = "left", 4
            ax.annotate(lbl, (x, y),
                        xytext=(xoff, 4), textcoords="offset points",
                        fontsize=7.5, color=color, ha=ha, va="bottom",
                        zorder=5)

    # Axes formatting
    ax.set_xlim(datetime(2024, 2, 1), datetime(2026, 9, 1))
    ax.set_ylim(0.38, 1.07)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xlabel("Model release date", fontsize=9)
    ax.set_ylabel("Cooperation rate (%)", fontsize=9)
    ax.set_title("PD Cooperation Rate Across Model Generations", fontsize=11, pad=8)

    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=8, framealpha=0.85, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for path in (SAVE_PDF, SAVE_PNG):
        path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 150 if path.suffix == ".png" else None
        fig.savefig(path, bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
