# game_theory_llm/analysis/visualization.py
"""All plotting code — reproduces every figure in the paper.

Figure map
----------
Fig 4   plot_cooperation_heatmaps         Cooperation% heatmap per model
Fig 5   plot_agreement_by_context         All-model agreement bar chart
Fig 6   plot_pairwise_agreement           Per-pair agreement matrices
Fig 7   plot_swap_distribution            Decision dist original vs swapped
Fig 8   plot_swap_delta_heatmaps          Cooperation-proportion change
Fig 9a  plot_defection_by_model           Defection rate bar chart (many models)
Fig 9b  plot_mmlu_vs_defection            MMLU score vs defection scatter
Fig 9c  plot_cramers_v_by_model           Cramer's V per model × category

Sharp-narrative figures
-----------------------
plot_cooperation_by_contrast_dim   Grouped bar: cooperation by contrast dim level × model
plot_game_model_heatmap            Heatmap: game_type × model cooperation rates
plot_contrast_effect_sizes         Grouped bar: Cramer's V effect sizes with significance stars
plot_swap_ablation_summary         Side-by-side bar: original vs swapped-aligned coop per model

Legacy helpers
--------------
plot_decision_distributions   (AnalysisResult-based, kept for compatibility)
plot_enhanced_visualizations  (wide DataFrame, original + swapped heatmaps)
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .._logging import get_logger
from ..models import AnalysisResult, PayoffMatrix

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Contrast dimensions used in the sharp-narrative design
# ---------------------------------------------------------------------------
_CONTRAST_DIMS = ("gender", "realism", "era", "contrast_domain", "observability")

_GAME_LABELS = {
    "prisoners_dilemma": "Prisoner's Dilemma",
    "stag_hunt": "Stag Hunt",
    "battle_of_the_sexes": "Battle of the Sexes",
    "chicken": "Chicken",
    "harmony": "Harmony",
    "deadlock": "Deadlock",
    "matching_pennies": "Matching Pennies",
}

_DIM_LABELS = {
    "gender": "Gender",
    "realism": "Realism",
    "era": "Era",
    "contrast_domain": "Contrast Domain",
    "observability": "Observability",
}

def _fmt_game(name: str) -> str:
    return _GAME_LABELS.get(name, name.replace("_", " ").title())

def _fmt_dim(name: str) -> str:
    return _DIM_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Consistent style helpers
# ---------------------------------------------------------------------------

# Tableau-inspired 8-colour palette used across all bar/line charts.
_PALETTE = [
    "#4E79A7",  # blue      — Game Type / primary series
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#9C755F",  # brown
    "#EDC948",  # yellow
    "#BAB0AC",  # gray
]


def _style_ax(ax, grid: bool = True) -> None:
    """Apply clean paper-ready style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    if grid:
        ax.yaxis.grid(True, color="#ebebeb", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _save(fig, path: Optional[str], name: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, name), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _auto_models(df: pd.DataFrame) -> List[str]:
    """Detect model names from ``decision_<model>`` columns in *df*."""
    return [c[len("decision_"):] for c in df.columns if c.startswith("decision_") and not c.startswith("decision_swapped_")]


def _resolve_models(df: pd.DataFrame, models: Optional[List[str]]) -> List[str]:
    """Return *models* if provided, else auto-detect from the DataFrame."""
    if models is not None:
        return models
    detected = _auto_models(df)
    if not detected:
        raise ValueError("No 'decision_<model>' columns found and no models list provided.")
    return detected


def _resolve_labels(
    models: List[str],
    model_labels: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Return a label mapping, falling back to the model name itself."""
    if model_labels is None:
        return {m: m for m in models}
    return {m: model_labels.get(m, m) for m in models}


# ===========================================================================
# Figure 4 — Cooperation heatmaps
# ===========================================================================

def plot_cooperation_heatmaps(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 4 — Per-model cooperation proportion by (game_type × contrast_dim).

    Produces one heatmap per model.
    Cooperation = ``decision_{model} == 'A'``.

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.
    models : list of str
        Model identifiers (matched against ``decision_{model}`` columns).
    model_labels : dict, optional
        Display names for each model.  Defaults to the model name itself.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting cooperation heatmaps (Fig 4)")
    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)
    col2 = "contrast_dim" if "contrast_dim" in df.columns else (
        "actor_type" if "actor_type" in df.columns else None
    )
    for model in models:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        df2 = df.copy()
        df2["cooperate"] = (df2[col] == "A").astype(float)

        if col2 and "game_type" in df2.columns:
            pivot = df2.groupby(["game_type", col2])["cooperate"].mean().unstack()
        elif "game_type" in df2.columns:
            pivot = df2.groupby("game_type")["cooperate"].mean().to_frame(name="overall")
        else:
            pivot = df2[["cooperate"]].T
        pivot.index = [_fmt_game(g) for g in pivot.index]
        pivot.columns = [_fmt_dim(c) for c in pivot.columns]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Cooperation Proportion — {labels[model]}")
        ax.set_xlabel("Contrast Dimension")
        ax.set_ylabel("Game Type")
        fig.tight_layout()
        _save(fig, save_dir, f"fig4_cooperation_heatmap_{model}.png")


# ===========================================================================
# Figure 5 — Agreement by context
# ===========================================================================

def plot_agreement_by_context(
    agreement_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 5 — All-model agreement bar chart with 95% CI error bars.

    Parameters
    ----------
    agreement_df : DataFrame
        Output of ``agreement.agreement_by_context()``.
        In the sharp-narrative design this is expected to contain columns
        ``contrast_dim``, ``game_type``, ``agreement``, ``ci_lower``,
        ``ci_upper``.  The function falls back gracefully to ``actor_type``
        and ``topic`` if those columns are present instead.
    """
    logger.info("Plotting agreement by context (Fig 5)")
    # Determine grouping columns available
    if "actor_type" in agreement_df.columns:
        group_col = "actor_type"
        item_col = "topic"
    else:
        group_col = "contrast_dim"
        item_col = "game_type"

    group_vals = agreement_df[group_col].unique()
    item_vals = sorted(agreement_df[item_col].unique())
    n_groups = len(group_vals)

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, grp in zip(axes, group_vals):
        sub = agreement_df[agreement_df[group_col] == grp].set_index(item_col)
        sub = sub.reindex(item_vals)
        y = sub["agreement"].values
        yerr_lo = y - sub["ci_lower"].values
        yerr_hi = sub["ci_upper"].values - y

        ax.bar(range(len(item_vals)), y, color=_PALETTE[0], alpha=0.88, edgecolor="none")
        ax.errorbar(
            range(len(item_vals)), y,
            yerr=[yerr_lo, yerr_hi],
            fmt="none", color="#333333", capsize=3, linewidth=1.0,
        )
        ax.set_xticks(range(len(item_vals)))
        ax.set_xticklabels([_fmt_game(v) for v in item_vals], rotation=40, ha="right", fontsize=9)
        ax.set_title(f"{_fmt_dim(grp)}", fontsize=11, pad=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Agreement (%)", fontsize=10)
        ax.axhline(0.5, color="#E15759", linestyle="--", linewidth=0.9, label="Chance")
        _style_ax(ax)

    fig.suptitle("All-Model Agreement by Context", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "fig5_agreement_by_context.png")


# ===========================================================================
# Figure 6 — Pairwise agreement matrices
# ===========================================================================

def plot_pairwise_agreement(
    pairwise_df: pd.DataFrame,
    context_label: str = "",
    save_dir: Optional[str] = None,
) -> None:
    """Fig 6 — Pairwise model agreement confusion matrices.

    Parameters
    ----------
    pairwise_df : DataFrame
        Output of ``agreement.pairwise_agreement()``.
    context_label : str
        Used in the figure title (e.g. "gender × prisoners_dilemma").
    """
    logger.info("Plotting pairwise agreement (Fig 6)")
    pairs = pairwise_df.index.tolist()
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        row = pairwise_df.loc[pair]
        mat = np.array([
            [row["m1_A_m2_A"], row["m1_A_m2_B"]],
            [row["m1_B_m2_A"], row["m1_B_m2_B"]],
        ])
        m1, m2 = pair.split("×")
        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"{m2}: A", f"{m2}: B"],
            yticklabels=[f"{m1}: A", f"{m1}: B"],
            vmin=0, vmax=1, ax=ax,
        )
        ax.set_title(f"{pair}\n(agree={row['agree']:.2f}, n={int(row['n'])})")

    title = f"Pairwise Agreement{' — ' + context_label if context_label else ''}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    safe_label = context_label.replace(" ", "_").replace("/", "-")
    _save(fig, save_dir, f"fig6_pairwise_agreement_{safe_label}.png")


# ===========================================================================
# Figure 7 — Swap distribution (original vs swapped per model)
# ===========================================================================

def plot_swap_distribution(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 7 — Decision distribution with original vs swapped label, per model.

    Shows the proportion choosing Cooperate in the original pass vs the
    swapped pass.  Cooperation = A (original) or B (swapped).

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting swap distribution (Fig 7)")
    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)
    rows = []
    for model in models:
        orig_col = f"decision_{model}"
        swap_col = f"decision_swapped_{model}"
        if orig_col not in df.columns:
            continue
        n = df[orig_col].notna().sum()
        coop_orig = (df[orig_col] == "A").sum() / n if n else float("nan")
        rows.append({"model": labels[model], "pass": "Original\n(A = Cooperate)", "cooperation": coop_orig})
        if swap_col in df.columns:
            n_sw = df[swap_col].notna().sum()
            coop_sw = (df[swap_col] == "B").sum() / n_sw if n_sw else float("nan")
            rows.append({"model": labels[model], "pass": "Swapped\n(B = Cooperate)", "cooperation": coop_sw})

    plot_df = pd.DataFrame(rows)
    models_present = [labels[m] for m in models if f"decision_{m}" in df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models_present))
    width = 0.35
    passes = plot_df["pass"].unique()
    for i, p in enumerate(passes):
        sub = plot_df[plot_df["pass"] == p]
        vals = [
            sub[sub["model"] == ml]["cooperation"].values[0]
            if (sub["model"] == ml).any() else float("nan")
            for ml in models_present
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, label=p, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.set_ylabel("Cooperation Proportion")
    ax.set_title("Cooperation Rate: Original vs Swapped Label (Fig 7)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save(fig, save_dir, "fig7_swap_distribution.png")


# ===========================================================================
# Figure 8 — Cooperation-proportion delta heatmaps
# ===========================================================================

def plot_swap_delta_heatmaps(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 8 — Change in cooperation proportion when Cooperate = B vs A.

    Delta = Pr(Cooperate | original) − Pr(Cooperate | swapped).
    Positive values → more cooperation when Cooperate is option A.
    Pivots on ``game_type × contrast_dim`` (sharp-narrative design).

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting swap delta heatmaps (Fig 8)")
    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)
    col2 = "contrast_dim" if "contrast_dim" in df.columns else (
        "actor_type" if "actor_type" in df.columns else None
    )
    for model in models:
        orig_col = f"decision_{model}"
        swap_col = f"decision_swapped_{model}"
        if orig_col not in df.columns or swap_col not in df.columns:
            continue

        df2 = df.copy()
        df2["coop_orig"] = (df2[orig_col] == "A").astype(float)
        df2["coop_swap"] = (df2[swap_col] == "B").astype(float)
        df2["delta"] = df2["coop_orig"] - df2["coop_swap"]

        if col2 and "game_type" in df2.columns:
            pivot = df2.groupby(["game_type", col2])["delta"].mean().unstack()
        elif "game_type" in df2.columns:
            pivot = df2.groupby("game_type")["delta"].mean().to_frame(name="overall")
        else:
            pivot = df2[["delta"]].mean().to_frame().T
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
        )
        ax.set_title(
            f"Cooperation Change (Original − Swapped) — {labels[model]}"
        )
        ax.set_xlabel("Contrast Dimension")
        ax.set_ylabel("Game Type")
        fig.tight_layout()
        _save(fig, save_dir, f"fig8_swap_delta_{model}.png")


# ===========================================================================
# Figure 9a — Defection rate by model (many models, Appendix D)
# ===========================================================================

def plot_defection_by_model(
    model_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 9a — Horizontal bar chart of defection rate across models.

    Parameters
    ----------
    model_df : DataFrame
        Output of ``predictive.cooperation_by_model()``.
        Must have columns ``model``, ``defection_rate``.
    """
    logger.info("Plotting defection by model (Fig 9a)")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(model_df) + 2))
    colors = ["#d62728" if r > 0.5 else "#1f77b4" for r in model_df["defection_rate"]]
    ax.barh(model_df["model"], model_df["defection_rate"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="50%")
    ax.set_xlabel("Defection Rate")
    ax.set_title("Defection Rate by Model (Appendix D)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, save_dir, "fig9a_defection_by_model.png")


# ===========================================================================
# Figure 9b — MMLU vs defection scatter (Appendix D)
# ===========================================================================

def plot_mmlu_vs_defection(
    model_df: pd.DataFrame,
    mmlu_col: str = "mmlu_score",
    defection_col: str = "defection_rate",
    family_col: Optional[str] = "family",
    save_dir: Optional[str] = None,
) -> None:
    """Fig 9b — Scatter of MMLU benchmark score vs defection rate.

    Parameters
    ----------
    model_df : DataFrame
        Must have *mmlu_col* and *defection_col* columns.
        Optionally *family_col* for colour-coding model families.
    """
    logger.info("Plotting MMLU vs defection (Fig 9b)")
    fig, ax = plt.subplots(figsize=(8, 6))

    if family_col and family_col in model_df.columns:
        families = model_df[family_col].unique()
        markers = ["o", "s", "^", "D", "v", "P", "*"]
        for i, fam in enumerate(families):
            sub = model_df[model_df[family_col] == fam]
            ax.scatter(
                sub[mmlu_col], sub[defection_col],
                label=fam, marker=markers[i % len(markers)], s=80,
            )
        ax.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.scatter(model_df[mmlu_col], model_df[defection_col], s=80, color="steelblue")

    # Annotate points
    for _, row in model_df.iterrows():
        ax.annotate(
            row.get("model", ""), (row[mmlu_col], row[defection_col]),
            fontsize=7, textcoords="offset points", xytext=(4, 4),
        )

    ax.set_xlabel("MMLU Score")
    ax.set_ylabel("Defection Rate")
    ax.set_title("Model Capability vs Defection Rate (Appendix D — Fig 9b)")
    ax.set_xlim(55, 95)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, save_dir, "fig9b_mmlu_vs_defection.png")


# ===========================================================================
# Figure 9c — Cramer's V per model × contextual category
# ===========================================================================

def plot_cramers_v_by_model(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 9c — Cramer's V effect size per model and contextual category.

    Computed directly from *df* for categories:
    ``game_type``, ``contrast_dim``, ``observability``, ``gender``,
    ``realism``, ``era``, ``contrast_domain``.

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting Cramer's V by model (Fig 9c)")
    models = _resolve_models(df, models)
    from scipy import stats

    categories = (
        "game_type", "gender", "realism", "era", "contrast_domain", "observability",
    )
    cat_labels = {
        "game_type": "Game Type",
        "gender": "Gender",
        "realism": "Realism",
        "era": "Era",
        "contrast_domain": "Contrast Domain",
        "observability": "Observability",
    }
    labels = _resolve_labels(models, model_labels)
    rows = []
    for model in models:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for cat in categories:
            if cat not in df.columns:
                continue
            ct = pd.crosstab(df[cat], df[col])
            if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, _, _, _ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            min_dim = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
            rows.append({"model": labels[model], "category": cat, "cramers_v": v})

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        logger.warning("No data to plot for plot_cramers_v_by_model")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = plot_df.pivot(index="model", columns="category", values="cramers_v")
    ordered_cats = [c for c in ("game_type", "gender", "realism", "era", "contrast_domain", "observability") if c in pivot.columns]
    pivot = pivot[ordered_cats]
    pivot.columns = [cat_labels.get(c, c) for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=_PALETTE[:len(pivot.columns)],
               width=0.72, edgecolor="none", alpha=0.90)
    _style_ax(ax)
    ax.set_ylabel("Cramér's V", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Effect Size (Cramér's V) by Model and Context Category",
                 fontsize=13, pad=10)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right", fontsize=10)
    max_v = pivot.values[~np.isnan(pivot.values)].max() if pivot.size else 1.0
    ax.set_ylim(0, min(1.0, max_v * 1.22 + 0.02))
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title_fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_dir, "fig9c_cramers_v_by_model.png")


# ===========================================================================
# Figure 10 — Game recognition (Appendix E)
# ===========================================================================

def plot_game_recognition(
    recognition_df: pd.DataFrame,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 10 — Cooperation proportion when game theory IS/IS NOT mentioned.

    Parameters
    ----------
    recognition_df : DataFrame
        Output of ``agreement.cooperation_by_recognition()``.
        Must have columns ``model``, ``recognized``, ``cooperation``.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting game recognition (Fig 10)")
    rec_models = recognition_df["model"].unique()
    fig, axes = plt.subplots(1, len(rec_models), figsize=(4 * len(rec_models), 5), sharey=True)
    if len(rec_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, rec_models):
        sub = recognition_df[recognition_df["model"] == model]
        display = model_labels.get(model, model) if model_labels else model
        labels = ["Mentioned\nGame Theory", "Did Not Mention"]
        vals = [
            sub[sub["recognized"] == True]["cooperation"].values[0]
            if (sub["recognized"] == True).any() else float("nan"),
            sub[sub["recognized"] == False]["cooperation"].values[0]
            if (sub["recognized"] == False).any() else float("nan"),
        ]
        colors = ["#ff7f0e", "#1f77b4"]
        ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(display)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Cooperation Proportion")

    fig.suptitle("Cooperation Rate by Game Theory Recognition (Fig 10)", fontsize=12)
    fig.tight_layout()
    _save(fig, save_dir, "fig10_game_recognition.png")


# ===========================================================================
# Focal-decision rate by game type (cross-game comparison)
# ===========================================================================

def plot_focal_rate_by_game(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart showing focal-decision rate per game type per model.

    The "focal decision" is always Decision A in canonical form.
    This is the key cross-game comparison figure.

    Parameters
    ----------
    df : DataFrame
        Must have ``game_type`` and ``decision_{model}`` columns.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting focal rate by game (cross-game comparison)")
    if "game_type" not in df.columns:
        logger.warning("No game_type column found, skipping plot_focal_rate_by_game")
        return

    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)
    game_types = sorted(df["game_type"].unique())
    rows = []
    for model in models:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for gt in game_types:
            sub = df[df["game_type"] == gt]
            n = sub[col].notna().sum()
            focal = (sub[col] == "A").sum() / n if n > 0 else float("nan")
            rows.append({
                "model": labels[model],
                "game_type": gt,
                "focal_rate": focal,
                "n": n,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(game_types))
    models_present = plot_df["model"].unique()
    n_models = len(models_present)
    width = 0.8 / max(n_models, 1)

    for i, model_label in enumerate(models_present):
        sub = plot_df[plot_df["model"] == model_label]
        vals = [
            sub[sub["game_type"] == gt]["focal_rate"].values[0]
            if (sub["game_type"] == gt).any() else float("nan")
            for gt in game_types
        ]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model_label,
               color=_PALETTE[i % len(_PALETTE)], alpha=0.90, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_game(gt) for gt in game_types], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Decision A Rate", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Decision A Rate by Game Type and Model", fontsize=13, pad=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#E15759", linestyle="--", linewidth=0.9, label="50%")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_dir, "focal_rate_by_game.png")


# ===========================================================================
# Single-turn vs multi-turn comparison
# ===========================================================================

def plot_single_vs_multi_turn(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing cooperation rate per model across conversation modes.

    Parameters
    ----------
    df : DataFrame
        Must have ``conversation_mode`` and ``decision_{model}`` columns.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting single-turn vs multi-turn comparison")
    if "conversation_mode" not in df.columns:
        logger.warning("No conversation_mode column found, skipping plot_single_vs_multi_turn")
        return

    modes = sorted(df["conversation_mode"].unique())
    if len(modes) < 2:
        logger.warning("Only one conversation mode present, skipping plot_single_vs_multi_turn")
        return

    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)
    rows = []
    for model in models:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for mode in modes:
            sub = df[df["conversation_mode"] == mode]
            n = sub[col].notna().sum()
            coop = (sub[col] == "A").sum() / n if n > 0 else float("nan")
            rows.append({
                "model": labels[model],
                "conversation_mode": mode,
                "cooperation": coop,
                "n": n,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    models_present = [labels[m] for m in models if f"decision_{m}" in df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models_present))
    width = 0.35

    for i, mode in enumerate(modes):
        sub = plot_df[plot_df["conversation_mode"] == mode]
        vals = [
            sub[sub["model"] == ml]["cooperation"].values[0]
            if (sub["model"] == ml).any() else float("nan")
            for ml in models_present
        ]
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=mode.replace("_", " ").title(), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models_present)
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Cooperation Rate: Single-Turn vs Multi-Turn")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.legend(title="Conversation Mode")
    fig.tight_layout()
    _save(fig, save_dir, "single_vs_multi_turn.png")


# ===========================================================================
# Sharp-narrative figures
# ===========================================================================

def plot_cooperation_by_contrast_dim(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Main framing-effect figure — cooperation rate by contrast dimension level × model.

    For each of the 5 binary contrast dimensions (gender, realism, era,
    contrast_domain, observability), plots a grouped bar chart where:
      - x-axis: the two levels of that dimension (e.g. "male" / "female")
      - bars: grouped by model

    All 5 subplots appear in a single figure.

    The active level for a row is read from the dim-named column
    (e.g. ``df["gender"]``), which is non-null only when ``contrast_dim ==
    "gender"``.

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.  Must have ``contrast_dim`` and
        one column per contrast dim containing the level value.
    models : list of str
        Model identifiers (matched against ``decision_{model}`` columns).
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting cooperation by contrast dimension")
    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)

    dims = [d for d in _CONTRAST_DIMS if d in df.columns]
    n_dims = len(dims)
    if n_dims == 0:
        logger.warning("No contrast dim columns found, skipping plot_cooperation_by_contrast_dim")
        return

    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        sub = df[df["contrast_dim"] == dim].copy() if "contrast_dim" in df.columns else df.copy()
        # levels for this dimension
        levels = sorted(sub[dim].dropna().unique())
        if not levels:
            ax.set_title(dim)
            continue

        x = np.arange(len(levels))
        present_models = [m for m in models if f"decision_{m}" in df.columns]
        n_models = len(present_models)
        width = 0.8 / max(n_models, 1)

        for i, model in enumerate(present_models):
            col = f"decision_{model}"
            vals = []
            for lv in levels:
                mask = sub[dim] == lv
                n = sub.loc[mask, col].notna().sum()
                rate = (sub.loc[mask, col] == "A").sum() / n if n > 0 else float("nan")
                vals.append(rate)
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=labels[model],
                   color=_PALETTE[i % len(_PALETTE)], alpha=0.90, edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels([str(lv).capitalize() for lv in levels], rotation=0, ha="center", fontsize=10)
        ax.set_title(_fmt_dim(dim), fontsize=11, pad=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#E15759", linestyle="--", linewidth=0.9)
        _style_ax(ax)
        if ax is axes[0]:
            ax.set_ylabel("Decision A Rate", fontsize=11)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, title="Model", loc="lower center",
               ncol=min(len(present_models), 7), bbox_to_anchor=(0.5, -0.06),
               fontsize=9, title_fontsize=9)
    fig.suptitle("Decision A Rate by Contrast Dimension", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "fig_cooperation_by_contrast_dim.png")


def plot_game_model_heatmap(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Heatmap of cooperation rate: rows = game_type, columns = models.

    Cell value = proportion of rows where ``decision_{model} == 'A'``.
    Annotated with the cooperation rate.  Uses RdYlGn colormap.

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame from ``load_run()``.
    models : list of str
        Model identifiers.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting game × model cooperation heatmap")
    models = _resolve_models(df, models)
    labels = _resolve_labels(models, model_labels)

    if "game_type" not in df.columns:
        logger.warning("No game_type column, skipping plot_game_model_heatmap")
        return

    game_types = sorted(df["game_type"].unique())
    present_models = [m for m in models if f"decision_{m}" in df.columns]
    if not present_models:
        logger.warning("No decision columns found, skipping plot_game_model_heatmap")
        return

    data = {}
    for model in present_models:
        col = f"decision_{model}"
        rates = []
        for gt in game_types:
            sub = df[df["game_type"] == gt]
            n = sub[col].notna().sum()
            rate = (sub[col] == "A").sum() / n if n > 0 else float("nan")
            rates.append(rate)
        data[labels[model]] = rates

    heatmap_df = pd.DataFrame(data, index=game_types)
    heatmap_df.index = [_fmt_game(g) for g in heatmap_df.index]

    fig, ax = plt.subplots(figsize=(max(6, len(present_models) * 1.4), max(4, len(game_types) * 0.8)))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title("Decision-A Rate by Game Type and Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Game Type")
    fig.tight_layout()
    _save(fig, save_dir, "fig_game_model_heatmap.png")


def plot_contrast_effect_sizes(
    stats_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart of Cramer's V effect sizes with significance stars.

    Takes the output of ``compute_contrast_statistics()`` (columns:
    ``model``, ``contrast_dim``, ``cramers_v``, ``p_corrected``).
    Plots bars grouped by model per contrast_dim.  Stars (*) are annotated
    above bars where ``p_corrected < 0.05``.

    Parameters
    ----------
    stats_df : DataFrame
        Must have columns ``model``, ``contrast_dim``, ``cramers_v``,
        ``p_corrected``.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting contrast effect sizes")
    required = {"model", "contrast_dim", "cramers_v", "p_corrected"}
    missing = required - set(stats_df.columns)
    if missing:
        logger.warning("Missing columns %s, skipping plot_contrast_effect_sizes", missing)
        return

    dims = sorted(stats_df["contrast_dim"].unique())
    model_list = list(stats_df["model"].unique())
    n_models = len(model_list)
    x = np.arange(len(dims))
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(dims) * 1.6), 5))

    for i, model in enumerate(model_list):
        sub = stats_df[stats_df["model"] == model]
        vals = []
        sigs = []
        for dim in dims:
            row = sub[sub["contrast_dim"] == dim]
            if row.empty:
                vals.append(float("nan"))
                sigs.append(False)
            else:
                vals.append(row["cramers_v"].values[0])
                sigs.append(row["p_corrected"].values[0] < 0.05)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=_PALETTE[i % len(_PALETTE)], alpha=0.90, edgecolor="none")

        for bar, sig, v in zip(bars, sigs, vals):
            if sig and not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.005,
                    "*",
                    ha="center", va="bottom", fontsize=12, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_dim(d) for d in dims], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Cramér's V", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Contrast Dimension Effect Sizes by Model  (* p < 0.05 corrected)",
                 fontsize=13, pad=10)
    ax.set_ylim(0, min(1.0, stats_df["cramers_v"].max() * 1.30 + 0.02))
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_dir, "fig_contrast_effect_sizes.png")


def plot_swap_ablation_summary(
    comparison_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Side-by-side bar chart: original vs swapped-aligned cooperation per model.

    Takes the output of ``swap_ablation_comparison()`` (index = model,
    columns: ``chose_a_original``, ``chose_a_swapped_aligned``, ``delta``,
    ``n``).  The delta is annotated above each pair of bars.

    Parameters
    ----------
    comparison_df : DataFrame
        Must have columns ``chose_a_original``, ``chose_a_swapped_aligned``,
        ``delta``.  Index (or a ``model`` column) identifies each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Plotting swap ablation summary")
    required = {"chose_a_original", "chose_a_swapped_aligned", "delta"}
    missing = required - set(comparison_df.columns)
    if missing:
        logger.warning("Missing columns %s, skipping plot_swap_ablation_summary", missing)
        return

    # Accept either index-as-model or explicit model column
    if "model" in comparison_df.columns:
        df_plot = comparison_df.set_index("model")
    else:
        df_plot = comparison_df.copy()

    model_names = list(df_plot.index)
    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    bars_orig = ax.bar(x - width / 2, df_plot["chose_a_original"], width,
                       label="Original", color="#1f77b4", alpha=0.85)
    bars_swap = ax.bar(x + width / 2, df_plot["chose_a_swapped_aligned"], width,
                       label="Swapped-Aligned", color="#ff7f0e", alpha=0.85)

    # Annotate delta
    for xi, model in zip(x, model_names):
        delta = df_plot.loc[model, "delta"]
        top = max(
            df_plot.loc[model, "chose_a_original"],
            df_plot.loc[model, "chose_a_swapped_aligned"],
        )
        sign = "+" if delta >= 0 else ""
        ax.text(xi, top + 0.025, f"{sign}{delta:.2f}", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Cooperation Rate (chose A)")
    ax.set_title("Swap Ablation: Original vs Swapped-Aligned Cooperation")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_dir, "fig_swap_ablation_summary.png")


# ===========================================================================
# Legacy helpers (kept for backward compatibility)
# ===========================================================================

def plot_decision_distributions(
    results: AnalysisResult,
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Bar charts of overall decision distribution per model (legacy).

    Parameters
    ----------
    results : AnalysisResult
        Legacy analysis result object.
    models : list of str, optional
        Model identifiers to plot.  Defaults to keys in ``results.decisions``.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str, optional
        Directory to write PNG files into.
    """
    logger.info("Creating decision distribution plots")
    if models is None:
        models = list(results.decisions.keys())
    labels = _resolve_labels(models, model_labels)
    for model in models:
        if model not in results.decisions:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.Series(results.decisions[model]).value_counts().plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"Overall Decision Distribution - {labels[model]}")
        ax.set_xlabel("Decision")
        ax.set_ylabel("Count")
        _save(fig, save_dir, f"overall_dist_{model}.png")
    logger.info("Decision distribution plots complete")


def plot_enhanced_visualizations(
    df: pd.DataFrame,
    payoff_matrix: Optional[PayoffMatrix],
    models: Optional[List[str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    save_dir: str = "analysis_results",
    include_swapped: bool = False,
) -> None:
    """Per-model EV curves, stacked bar charts, and heatmaps (legacy).

    Parameters
    ----------
    df : DataFrame
        Wide DataFrame.
    payoff_matrix : PayoffMatrix or None
        Payoff matrix for EV curve computation.
    models : list of str, optional
        Model identifiers.  Defaults to inferring from ``decision_*`` columns.
    model_labels : dict, optional
        Display names for each model.
    save_dir : str
        Base directory for outputs.
    include_swapped : bool
        Whether to also plot swapped-label figures.
    """
    logger.info("Generating enhanced visualizations")

    if models is None:
        models = [
            c.removeprefix("decision_")
            for c in df.columns
            if c.startswith("decision_") and not c.startswith("decision_swapped_")
        ]
    labels = _resolve_labels(models, model_labels)

    for model in models:
        model_dir = os.path.join(save_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        mdf = df.copy()
        mdf["decision"] = df.get(f"decision_{model}")
        decision_cols = ["decision"]
        if include_swapped and f"decision_swapped_{model}" in df.columns:
            mdf["decision_swapped"] = df[f"decision_swapped_{model}"]
            decision_cols.append("decision_swapped")

        # EV curve
        if payoff_matrix is not None:
            for dt in decision_cols:
                p_range = np.linspace(0, 1, 100)

                def _ev(p, m=payoff_matrix):
                    pb = 1 - p
                    return (
                        p * p * m.matrix[0][0] + p * pb * m.matrix[1][0]
                        + pb * p * m.matrix[2][0] + pb * pb * m.matrix[3][0]
                    )

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(p_range, [_ev(p) for p in p_range])
                suffix = " (Swapped)" if dt == "decision_swapped" else ""
                ax.set_title(f"Expected Utility vs P(A) — {labels[model]}{suffix}")
                ax.set_xlabel("Probability of choosing A")
                ax.set_ylabel("Expected Utility")
                tag = "_swapped" if dt == "decision_swapped" else ""
                _save(fig, model_dir, f"expected_value_analysis{tag}.png")

        # Stacked bars
        for category in ("topic", "actor_type", "game_type", "contrast_dim", "observability", "conversation_mode"):
            for dt in decision_cols:
                if dt not in mdf.columns or category not in mdf.columns:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                ct = pd.crosstab(mdf[category], mdf[dt])
                (ct.div(ct.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax)
                suffix = " (Swapped)" if dt == "decision_swapped" else ""
                ax.set_title(f"Decision by {category} — {labels[model]}{suffix}")
                ax.set_xlabel(category)
                ax.set_ylabel("Percentage")
                fig.tight_layout()
                _save(fig, model_dir, f"decision_by_{category}_{dt}.png")

        # Swapped extras
        if include_swapped and "decision_swapped" in mdf.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            (mdf["decision"] != mdf["decision_swapped"]).value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Decision Changes After Swapping — {labels[model]}")
            ax.set_xlabel("Decision Changed")
            ax.set_ylabel("Count")
            fig.tight_layout()
            _save(fig, model_dir, "decision_changes.png")

            fig, ax = plt.subplots(figsize=(8, 6))
            cm = pd.crosstab(mdf["decision"], mdf["decision_swapped"], normalize="all") * 100
            sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"Decision Consistency Matrix — {labels[model]}")
            ax.set_xlabel("Swapped Decision")
            ax.set_ylabel("Original Decision")
            fig.tight_layout()
            _save(fig, model_dir, "decision_consistency_matrix.png")

        # Heatmaps
        for dt in decision_cols:
            if dt not in mdf.columns:
                continue
            # Use new grouping dims if available, else fall back to old
            row_col = "game_type" if "game_type" in mdf.columns else "topic"
            col_col = "contrast_dim" if "contrast_dim" in mdf.columns else "actor_type"
            if row_col not in mdf.columns or col_col not in mdf.columns:
                continue
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot = pd.crosstab(
                [mdf[row_col], mdf["observability"]] if "observability" in mdf.columns else mdf[row_col],
                [mdf[col_col], mdf[dt]],
            )
            sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            suffix = " (Swapped)" if dt == "decision_swapped" else ""
            ax.set_title(f"Decision Heatmap — {labels[model]}{suffix}")
            fig.tight_layout()
            _save(fig, model_dir, f"decision_heatmap_{dt}.png")

    logger.info("Enhanced visualizations complete")


def plot_cross_game_focal_rate(df, title="Focal-A rate by game"):
    """Bar plot of focal-A rate per game.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``cross_game_focal_rate_table`` (columns: game_type, n, focal_a_rate).
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("focal_a_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_sorted["game_type"], df_sorted["focal_a_rate"], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Focal-A rate")
    ax.set_xlabel("Game")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig
