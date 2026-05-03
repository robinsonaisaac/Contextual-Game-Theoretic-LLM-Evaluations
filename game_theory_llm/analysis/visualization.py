# game_theory_llm/analysis/visualization.py
"""All plotting code — reproduces every figure in the paper.

Figure map
----------
Fig 4   plot_cooperation_heatmaps         Cooperation% heatmap per model
Fig 5   plot_agreement_by_context         All-3-model agreement bar chart
Fig 6   plot_pairwise_agreement           Per-pair agreement matrices
Fig 7   plot_swap_distribution            Decision dist original vs swapped
Fig 8   plot_swap_delta_heatmaps          Cooperation-proportion change
Fig 9a  plot_defection_by_model           Defection rate bar chart (many models)
Fig 9b  plot_mmlu_vs_defection            MMLU score vs defection scatter
Fig 9c  plot_cramers_v_by_model           Cramer's V per model × category
Fig 10  plot_game_recognition             Cooperation by game-theory mention

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

_MODELS = ("llama", "claude", "gpt4")
_MODEL_LABELS = {"llama": "Llama", "claude": "Claude", "gpt4": "GPT-4o"}


def _save(fig, path: Optional[str], name: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, name), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ===========================================================================
# Figure 4 — Cooperation heatmaps
# ===========================================================================

def plot_cooperation_heatmaps(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 4 — Per-model cooperation proportion by (topic × actor type).

    Produces one heatmap per model, faceted by world type.
    Cooperation = ``decision_{model} == 'A'``.
    """
    logger.info("Plotting cooperation heatmaps (Fig 4)")
    for model in _MODELS:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        df2 = df.copy()
        df2["cooperate"] = (df2[col] == "A").astype(float)

        pivot = df2.groupby(["topic", "actor_type"])["cooperate"].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Cooperation Proportion — {_MODEL_LABELS[model]}")
        ax.set_xlabel("Actor Type")
        ax.set_ylabel("Topic")
        fig.tight_layout()
        _save(fig, save_dir, f"fig4_cooperation_heatmap_{model}.png")


# ===========================================================================
# Figure 5 — Agreement by topic × actor type
# ===========================================================================

def plot_agreement_by_context(
    agreement_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 5 — All-3-model agreement bar chart with 95% CI error bars.

    Parameters
    ----------
    agreement_df : DataFrame
        Output of ``agreement.agreement_by_context()``.
        Must have columns: ``topic``, ``actor_type``, ``agreement``,
        ``ci_lower``, ``ci_upper``.
    """
    logger.info("Plotting agreement by context (Fig 5)")
    actor_types = agreement_df["actor_type"].unique()
    topics = sorted(agreement_df["topic"].unique())
    n_actors = len(actor_types)

    fig, axes = plt.subplots(1, n_actors, figsize=(6 * n_actors, 6), sharey=True)
    if n_actors == 1:
        axes = [axes]

    for ax, actor in zip(axes, actor_types):
        sub = agreement_df[agreement_df["actor_type"] == actor].set_index("topic")
        sub = sub.reindex(topics)
        y = sub["agreement"].values
        yerr_lo = y - sub["ci_lower"].values
        yerr_hi = sub["ci_upper"].values - y

        ax.bar(range(len(topics)), y, color="steelblue", alpha=0.8)
        ax.errorbar(
            range(len(topics)), y,
            yerr=[yerr_lo, yerr_hi],
            fmt="none", color="black", capsize=4,
        )
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Actor: {actor}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Agreement (%)")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Chance")

    fig.suptitle("All-Model Agreement by Topic and Actor Type", fontsize=13)
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
        Used in the figure title (e.g. "Allies × 21st C. Global Politics").
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
    save_dir: Optional[str] = None,
) -> None:
    """Fig 7 — Decision distribution with original vs swapped label, per model.

    Shows the proportion choosing Cooperate in the original pass vs the
    swapped pass.  Cooperation = A (original) or B (swapped).
    """
    logger.info("Plotting swap distribution (Fig 7)")
    rows = []
    for model in _MODELS:
        orig_col = f"decision_{model}"
        swap_col = f"decision_swapped_{model}"
        if orig_col not in df.columns:
            continue
        n = df[orig_col].notna().sum()
        coop_orig = (df[orig_col] == "A").sum() / n if n else float("nan")
        rows.append({"model": _MODEL_LABELS[model], "pass": "Original\n(A = Cooperate)", "cooperation": coop_orig})
        if swap_col in df.columns:
            n_sw = df[swap_col].notna().sum()
            coop_sw = (df[swap_col] == "B").sum() / n_sw if n_sw else float("nan")
            rows.append({"model": _MODEL_LABELS[model], "pass": "Swapped\n(B = Cooperate)", "cooperation": coop_sw})

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(_MODELS))
    width = 0.35
    passes = plot_df["pass"].unique()
    for i, p in enumerate(passes):
        sub = plot_df[plot_df["pass"] == p]
        vals = [sub[sub["model"] == _MODEL_LABELS[m]]["cooperation"].values[0]
                if (sub["model"] == _MODEL_LABELS[m]).any() else float("nan")
                for m in _MODELS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=p, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABELS[m] for m in _MODELS])
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
    save_dir: Optional[str] = None,
) -> None:
    """Fig 8 — Change in cooperation proportion when Cooperate = B vs A.

    Delta = Pr(Cooperate | original) − Pr(Cooperate | swapped).
    Positive values → more cooperation when Cooperate is option A.
    """
    logger.info("Plotting swap delta heatmaps (Fig 8)")
    for model in _MODELS:
        orig_col = f"decision_{model}"
        swap_col = f"decision_swapped_{model}"
        if orig_col not in df.columns or swap_col not in df.columns:
            continue

        df2 = df.copy()
        df2["coop_orig"] = (df2[orig_col] == "A").astype(float)
        df2["coop_swap"] = (df2[swap_col] == "B").astype(float)
        df2["delta"] = df2["coop_orig"] - df2["coop_swap"]

        pivot = df2.groupby(["topic", "actor_type"])["delta"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
        )
        ax.set_title(
            f"Cooperation Change (Original − Swapped) — {_MODEL_LABELS[model]}"
        )
        ax.set_xlabel("Actor Type")
        ax.set_ylabel("Topic")
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
    save_dir: Optional[str] = None,
) -> None:
    """Fig 9c — Cramer's V effect size per model and contextual category.

    Computed directly from *df* for categories topic, actor_type, observability, power_dynamic.
    """
    logger.info("Plotting Cramer's V by model (Fig 9c)")
    from scipy import stats

    categories = ("topic", "actor_type", "observability", "power_dynamic", "game_type", "conversation_mode")
    rows = []
    for model in _MODELS:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for cat in categories:
            ct = pd.crosstab(df[cat], df[col])
            chi2, _, _, _ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            min_dim = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
            rows.append({"model": _MODEL_LABELS[model], "category": cat, "cramers_v": v})

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = plot_df.pivot(index="model", columns="category", values="cramers_v")
    pivot.plot(kind="bar", ax=ax, alpha=0.85)
    ax.set_ylabel("Cramer's V")
    ax.set_title("Effect Size (Cramer's V) per Model and Category (Fig 9c)")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Category")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, save_dir, "fig9c_cramers_v_by_model.png")


# ===========================================================================
# Figure 10 — Game recognition (Appendix E)
# ===========================================================================

def plot_game_recognition(
    recognition_df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Fig 10 — Cooperation proportion when game theory IS/IS NOT mentioned.

    Parameters
    ----------
    recognition_df : DataFrame
        Output of ``agreement.cooperation_by_recognition()``.
        Must have columns ``model``, ``recognized``, ``cooperation``.
    """
    logger.info("Plotting game recognition (Fig 10)")
    models = recognition_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = recognition_df[recognition_df["model"] == model]
        labels = ["Mentioned\nGame Theory", "Did Not Mention"]
        vals = [
            sub[sub["recognized"] == True]["cooperation"].values[0]
            if (sub["recognized"] == True).any() else float("nan"),
            sub[sub["recognized"] == False]["cooperation"].values[0]
            if (sub["recognized"] == False).any() else float("nan"),
        ]
        colors = ["#ff7f0e", "#1f77b4"]
        ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(_MODEL_LABELS.get(model, model))
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
    save_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart showing focal-decision rate per game type per model.

    The "focal decision" is always Decision A in canonical form.
    This is the key cross-game comparison figure.

    Parameters
    ----------
    df : DataFrame
        Must have ``game_type`` and ``decision_{model}`` columns.
    """
    logger.info("Plotting focal rate by game (cross-game comparison)")
    if "game_type" not in df.columns:
        logger.warning("No game_type column found, skipping plot_focal_rate_by_game")
        return

    game_types = sorted(df["game_type"].unique())
    rows = []
    for model in _MODELS:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for gt in game_types:
            sub = df[df["game_type"] == gt]
            n = sub[col].notna().sum()
            focal = (sub[col] == "A").sum() / n if n > 0 else float("nan")
            rows.append({
                "model": _MODEL_LABELS.get(model, model),
                "game_type": gt,
                "focal_rate": focal,
                "n": n,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(game_types))
    n_models = len([m for m in _MODELS if f"decision_{m}" in df.columns])
    width = 0.8 / max(n_models, 1)

    models_present = plot_df["model"].unique()
    for i, model_label in enumerate(models_present):
        sub = plot_df[plot_df["model"] == model_label]
        vals = [
            sub[sub["game_type"] == gt]["focal_rate"].values[0]
            if (sub["game_type"] == gt).any() else float("nan")
            for gt in game_types
        ]
        offset = (i - (len(models_present) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model_label, alpha=0.85)

    ax.set_xticks(x)
    # Format game type labels nicely
    labels = [gt.replace("_", " ").title() for gt in game_types]
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Focal Decision (A) Rate")
    ax.set_title("Focal Decision Rate by Game Type and Model")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="50%")
    ax.legend(title="Model")
    fig.tight_layout()
    _save(fig, save_dir, "focal_rate_by_game.png")


# ===========================================================================
# Single-turn vs multi-turn comparison
# ===========================================================================

def plot_single_vs_multi_turn(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing cooperation rate per model across conversation modes.

    Parameters
    ----------
    df : DataFrame
        Must have ``conversation_mode`` and ``decision_{model}`` columns.
    """
    logger.info("Plotting single-turn vs multi-turn comparison")
    if "conversation_mode" not in df.columns:
        logger.warning("No conversation_mode column found, skipping plot_single_vs_multi_turn")
        return

    modes = sorted(df["conversation_mode"].unique())
    if len(modes) < 2:
        logger.warning("Only one conversation mode present, skipping plot_single_vs_multi_turn")
        return

    rows = []
    for model in _MODELS:
        col = f"decision_{model}"
        if col not in df.columns:
            continue
        for mode in modes:
            sub = df[df["conversation_mode"] == mode]
            n = sub[col].notna().sum()
            coop = (sub[col] == "A").sum() / n if n > 0 else float("nan")
            rows.append({
                "model": _MODEL_LABELS.get(model, model),
                "conversation_mode": mode,
                "cooperation": coop,
                "n": n,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len([m for m in _MODELS if f"decision_{m}" in df.columns]))
    width = 0.35
    models_present = [_MODEL_LABELS[m] for m in _MODELS if f"decision_{m}" in df.columns]

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
# Legacy helpers (kept for backward compatibility)
# ===========================================================================

def plot_decision_distributions(
    results: AnalysisResult,
    save_dir: Optional[str] = None,
) -> None:
    """Bar charts of overall decision distribution per model (legacy)."""
    logger.info("Creating decision distribution plots")
    for model in _MODELS:
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.Series(results.decisions[model]).value_counts().plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"Overall Decision Distribution - {_MODEL_LABELS[model]}")
        ax.set_xlabel("Decision")
        ax.set_ylabel("Count")
        _save(fig, save_dir, f"overall_dist_{model}.png")
    logger.info("Decision distribution plots complete")


def plot_enhanced_visualizations(
    df: pd.DataFrame,
    payoff_matrix: Optional[PayoffMatrix],
    save_dir: str = "analysis_results",
    include_swapped: bool = False,
) -> None:
    """Per-model EV curves, stacked bar charts, and heatmaps."""
    logger.info("Generating enhanced visualizations")

    for model in _MODELS:
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
                ax.set_title(f"Expected Utility vs P(A) — {_MODEL_LABELS[model]}{suffix}")
                ax.set_xlabel("Probability of choosing A")
                ax.set_ylabel("Expected Utility")
                tag = "_swapped" if dt == "decision_swapped" else ""
                _save(fig, model_dir, f"expected_value_analysis{tag}.png")

        # Stacked bars
        for category in ("topic", "actor_type", "observability", "power_dynamic", "conversation_mode"):
            for dt in decision_cols:
                if dt not in mdf.columns:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                ct = pd.crosstab(mdf[category], mdf[dt])
                (ct.div(ct.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax)
                suffix = " (Swapped)" if dt == "decision_swapped" else ""
                ax.set_title(f"Decision by {category} — {_MODEL_LABELS[model]}{suffix}")
                ax.set_xlabel(category)
                ax.set_ylabel("Percentage")
                fig.tight_layout()
                _save(fig, model_dir, f"decision_by_{category}_{dt}.png")

        # Swapped extras
        if include_swapped and "decision_swapped" in mdf.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            (mdf["decision"] != mdf["decision_swapped"]).value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Decision Changes After Swapping — {_MODEL_LABELS[model]}")
            ax.set_xlabel("Decision Changed")
            ax.set_ylabel("Count")
            fig.tight_layout()
            _save(fig, model_dir, "decision_changes.png")

            fig, ax = plt.subplots(figsize=(8, 6))
            cm = pd.crosstab(mdf["decision"], mdf["decision_swapped"], normalize="all") * 100
            sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"Decision Consistency Matrix — {_MODEL_LABELS[model]}")
            ax.set_xlabel("Swapped Decision")
            ax.set_ylabel("Original Decision")
            fig.tight_layout()
            _save(fig, model_dir, "decision_consistency_matrix.png")

        # Heatmaps
        for dt in decision_cols:
            if dt not in mdf.columns:
                continue
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot = pd.crosstab(
                [mdf["topic"], mdf["observability"]],
                [mdf["actor_type"], mdf[dt]],
            )
            sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            suffix = " (Swapped)" if dt == "decision_swapped" else ""
            ax.set_title(f"Decision Heatmap — {_MODEL_LABELS[model]}{suffix}")
            fig.tight_layout()
            _save(fig, model_dir, f"decision_heatmap_{dt}.png")

    logger.info("Enhanced visualizations complete")


def plot_focal_rate_by_game(df, title="Focal-A rate by game"):
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
