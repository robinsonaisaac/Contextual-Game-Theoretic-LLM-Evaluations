"""Figure: relationship framing shifts cooperation only where the game is ambiguous.

Reads the two production evaluation roots through the project's own analysis
pipeline (so cooperation is derived from ``focal_decision_label`` and the
canonical/swapped counterbalancing is respected), then renders a two-panel
dumbbell figure.

    python3 scripts/figure_relationship_framing.py [--min-share 0.90]

Panel A is the finding: the allies-to-enemies gap per game, sorted by size.
Panel B is the robustness check: the same gap within each topic, showing the
effect travels with the relational framing rather than the surface domain.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# A sibling worktree also ships game_theory_llm; put this repo first so the
# figure is built from the same code that produced the runs.
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from game_theory_llm import naturalistic_analysis as na
CORPUS = REPO / "data/runs/2026-07-28-naturalistic-six-game-context-production-opus-v5"
RUN_ROOTS = (
    REPO / "data/runs/2026-07-28-naturalistic-six-game-context-evaluation-production-v4",
    REPO
    / "data/runs/2026-07-28-naturalistic-six-game-context-evaluation-production-v5-openai",
)
# The paper includes figures by a path relative to its own directory, so write
# there rather than to the repo-root ResultsFigures/.
OUT_DIR = REPO / "-NEURIPS-2026-Framing-The-Game" / "ResultsFigures"

# Diverging encoding: allies and enemies are opposite poles of one ordered
# scale, neutral is its midpoint. Validated on the light chart surface
# (#fcfcfb): poles CVD dE 21.6, normal-vision dE 32.3, both >= 3:1 contrast.
# The midpoint is deliberately gray, and every mark is direct-labelled so
# identity never rests on hue alone.
ALLIES = "#2a78d6"
ENEMIES = "#e34948"
NEUTRAL = "#898781"
CONNECTOR = "#c3c2b7"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"

GAME_LABELS = {
    "stag_hunt": "Stag hunt",
    "prisoners_dilemma": "Prisoner's dilemma",
    "chicken": "Chicken",
    "battle_of_the_sexes": "Battle of the sexes",
    "harmony": "Harmony",
    "deadlock": "Deadlock",
}
TOPIC_LABELS = {
    "global_politics_21c": "Global politics",
    "politics": "Politics",
    "us_business": "US business",
    "sporting_events": "Sporting events",
}
# Structural note per game, so a reader can see why the gap is where it is.
GAME_NOTE = {
    "stag_hunt": "assurance",
    "prisoners_dilemma": "social dilemma",
    "chicken": "anti-coordination",
    "battle_of_the_sexes": "coordination",
    "harmony": "cooperate dominant",
    "deadlock": "defect dominant",
}


def load_rows() -> pd.DataFrame:
    _, _, stories, prompts, _ = na.load_corpus(CORPUS)
    frames, models = [], []
    for root in RUN_ROOTS:
        evaluations, _ = na.load_evaluations(root)
        if len(evaluations):
            frames.append(evaluations)
            models += na._discover_models(root)
    rows, _, _ = na.build_analysis_rows(
        prompts, stories, models, pd.concat(frames, ignore_index=True)
    )
    return rows


def prepare(rows: pd.DataFrame, min_share: float):
    valid = rows[rows["valid"] == True].copy()  # noqa: E712 - pandas mask
    valid["coop"] = valid["semantic_focal_choice"].astype(float)
    # Same rule as scripts/stats_relationship_framing.py: keep a model when it
    # answered essentially everything dispatched to it. A threshold on absolute
    # valid rows would drop exactly the models that refuse most.
    share = valid.groupby("model").size() / valid.groupby("model")["prompt_id"].nunique()
    complete = sorted(share[share >= min_share].index)
    return valid[valid["model"].isin(complete)], complete


def _dumbbell(ax, frame, order, labels, notes=None, show_note=True):
    """One row per key: neutral tick between the two poles, gap annotated."""
    y = range(len(order))
    for i, key in enumerate(order):
        a = frame.loc[key, "allies"]
        n = frame.loc[key, "neutral"]
        e = frame.loc[key, "enemies"]
        ax.plot([e, a], [i, i], color=CONNECTOR, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([e], [i], s=86, color=ENEMIES, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([a], [i], s=86, color=ALLIES, zorder=4, edgecolor=SURFACE, linewidth=1.5)
        # Neutral rides above the poles so it stays visible when a row collapses.
        ax.scatter([n], [i], s=52, color=NEUTRAL, zorder=5, marker="|", linewidths=2.2)
        gap = a - e
        # When all three conditions land within a couple of points the marks
        # overlap and the row reads as a single dot; say so rather than let the
        # reader infer one condition is missing.
        if max(a, n, e) - min(a, n, e) < 2.0:
            # Flip to the left when the cluster sits in the right half, so the
            # note never runs into the gap column.
            right_half = (a + n + e) / 3 > 50
            ax.annotate(
                f"all three ≈ {(a + n + e) / 3:.0f}%",
                xy=(min(a, n, e) if right_half else max(a, n, e), i),
                xytext=(-12 if right_half else 12, 0), textcoords="offset points",
                va="center", ha="right" if right_half else "left",
                fontsize=8.5, color=MUTED,
            )
        ax.text(
            103, i, f"{gap:+.1f}", va="center", ha="left",
            fontsize=9.5, color=INK if abs(gap) >= 10 else MUTED,
            fontweight="bold" if abs(gap) >= 10 else "normal",
        )
    ax.set_yticks(list(y))
    ticklabels = []
    for key in order:
        if show_note and notes:
            ticklabels.append(f"{labels[key]}\n{notes[key]}")
        else:
            ticklabels.append(labels[key])
    ax.set_yticklabels(ticklabels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(CONNECTOR)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, length=0)


def build(rows: pd.DataFrame, complete, out_stem: Path, partial_note: str):
    by_game = (
        rows.pivot_table(index="game_id", columns="relationship", values="coop", aggfunc="mean") * 100
    )
    by_game["gap"] = by_game["allies"] - by_game["enemies"]
    game_order = by_game.sort_values("gap", ascending=False).index.tolist()

    by_topic = (
        rows.pivot_table(index="topic_id", columns="relationship", values="coop", aggfunc="mean") * 100
    )
    by_topic["gap"] = by_topic["allies"] - by_topic["enemies"]
    topic_order = by_topic.sort_values("gap", ascending=False).index.tolist()

    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 8.4), height_ratios=[6, 4], facecolor=SURFACE
    )
    for ax in axes:
        ax.set_facecolor(SURFACE)

    _dumbbell(axes[0], by_game, game_order, GAME_LABELS, GAME_NOTE, show_note=True)
    _dumbbell(axes[1], by_topic, topic_order, TOPIC_LABELS, None, show_note=False)

    overall = rows.groupby("relationship")["coop"].mean() * 100
    fig.text(
        0.012, 0.975,
        "Relational framing moves choices only where the game is ambiguous",
        fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top",
    )
    fig.text(
        0.012, 0.943,
        f"Decision-A rate by partner framing. Overall {overall['allies']:.1f}% as allies "
        f"vs {overall['enemies']:.1f}% as enemies ({overall['allies']-overall['enemies']:+.1f} pp).",
        fontsize=10.5, color=INK_2, ha="left", va="top",
    )
    axes[0].set_title(
        "By game  —  gap collapses when one action dominates",
        fontsize=11, color=INK, loc="left", pad=12, fontweight="bold",
    )
    axes[1].set_title(
        "By topic  —  the same gap in every domain, so it is not surface content",
        fontsize=11, color=INK, loc="left", pad=12, fontweight="bold",
    )
    axes[0].annotate(
        "gap (pp)", xy=(103, 0), xytext=(0, 26), textcoords="offset points",
        fontsize=9, color=MUTED, ha="left", va="center",
    )

    handles = [
        Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=ALLIES,
               markeredgecolor=SURFACE, label="Allies"),
        Line2D([], [], marker="|", ls="", markersize=9, markeredgewidth=2,
               color=NEUTRAL, label="Neutral"),
        Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=ENEMIES,
               markeredgecolor=SURFACE, label="Enemies"),
    ]
    # Figure-level legend under the subtitle keeps it clear of both panel titles.
    legend = fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925),
        ncol=3, frameon=False, fontsize=10, handletextpad=0.4, columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)

    fig.text(
        0.012, 0.014,
        f"n = {len(rows):,} valid decisions from {len(complete)} models with complete coverage."
        f"\nDecision A = the focal action, i.e. the cooperative or group-optimal one where that"
        f"\ndistinction applies; canonical/swapped label order counterbalanced."
        f"{partial_note}",
        fontsize=8.5, color=MUTED, ha="left", va="bottom", linespacing=1.5,
    )
    fig.subplots_adjust(left=0.21, right=0.90, top=0.855, bottom=0.10, hspace=0.34)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=220, facecolor=SURFACE)
        print("wrote", path)
    plt.close(fig)
    return by_game, by_topic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-share", type=float, default=0.90)
    parser.add_argument("--out", default=str(OUT_DIR / "relationship_framing"))
    args = parser.parse_args()

    rows = load_rows()
    frame, complete = prepare(rows, args.min_share)
    total = rows["model"].nunique()
    partial = total - len(complete)
    note = (
        f" {partial} of {total} models excluded for incomplete coverage."
        if partial
        else " All 19 panel models complete."
    )
    by_game, by_topic = build(frame, complete, Path(args.out), note)
    print("\nby game:\n", by_game.round(1).to_string())
    print("\nby topic:\n", by_topic.round(1).to_string())


if __name__ == "__main__":
    main()
