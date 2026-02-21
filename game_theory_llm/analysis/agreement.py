# game_theory_llm/analysis/agreement.py
"""Inter-model agreement metrics and game-recognition analysis.

Produces the data behind:
  - Figure 5  (agreement by topic × actor type)
  - Figure 6  (pairwise agreement matrices)
  - Figure 10 (cooperation conditioned on game-theory recognition)
  - Fleiss' Kappa reported in the paper body

The DataFrame passed to these functions must have columns
``decision_llama``, ``decision_claude``, ``decision_gpt4``.

Cooperation convention (matches ``base.py``):
  - ``decision_{model} == 'A'``  → Cooperate
  - ``decision_swapped_{model} == 'B'``  → Cooperate (swapped pass)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._logging import get_logger

logger = get_logger(__name__)

_MODELS = ("llama", "claude", "gpt4")
_MODEL_PAIRS: List[Tuple[str, str]] = [
    ("llama", "claude"),
    ("llama", "gpt4"),
    ("claude", "gpt4"),
]


# ---------------------------------------------------------------------------
# All-model agreement
# ---------------------------------------------------------------------------

def all_model_agreement(df: pd.DataFrame) -> float:
    """Fraction of rows where all three models make the same decision.

    Parameters
    ----------
    df : DataFrame
        Must contain ``decision_llama``, ``decision_claude``, ``decision_gpt4``.
    """
    agree = (
        (df["decision_llama"] == df["decision_claude"])
        & (df["decision_claude"] == df["decision_gpt4"])
    )
    return agree.mean()


def agreement_by_context(
    df: pd.DataFrame,
    group_cols: List[str] = ("topic", "actor_type"),
) -> pd.DataFrame:
    """All-model agreement and 95% CI grouped by *group_cols*.

    Returns a DataFrame with columns *group_cols*, ``agreement``,
    ``ci_lower``, ``ci_upper``, and ``n``.
    """
    rows = []
    for keys, grp in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(grp)
        p = all_model_agreement(grp)
        # Wilson score CI (works well for proportions near 0/1)
        z = 1.96
        ci_half = z * np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        row = dict(zip(group_cols, keys))
        row.update({"agreement": p, "ci_lower": max(0, p - ci_half),
                    "ci_upper": min(1, p + ci_half), "n": n})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------

def pairwise_agreement(
    df: pd.DataFrame,
    topic: Optional[str] = None,
    actor_type: Optional[str] = None,
    observability: Optional[str] = None,
    power_dynamic: Optional[str] = None,
) -> pd.DataFrame:
    """Per-pair agreement fractions, optionally filtered by context.

    Returns a DataFrame indexed by model pair with columns
    ``agree``, ``m1_A_m2_A``, ``m1_A_m2_B``, ``m1_B_m2_A``, ``m1_B_m2_B``,
    ``n``.

    These are the per-pair 2x2 agreement matrices as shown in Figure 6.
    """
    sub = df.copy()
    if topic is not None:
        sub = sub[sub["topic"] == topic]
    if actor_type is not None:
        sub = sub[sub["actor_type"] == actor_type]
    if observability is not None:
        sub = sub[sub["observability"] == observability]
    if power_dynamic is not None:
        sub = sub[sub["power_dynamic"] == power_dynamic]

    rows = []
    for m1, m2 in _MODEL_PAIRS:
        c1 = f"decision_{m1}"
        c2 = f"decision_{m2}"
        valid = sub[[c1, c2]].dropna()
        n = len(valid)
        if n == 0:
            rows.append({
                "pair": f"{m1}×{m2}", "agree": float("nan"),
                "m1_A_m2_A": 0, "m1_A_m2_B": 0,
                "m1_B_m2_A": 0, "m1_B_m2_B": 0, "n": 0,
            })
            continue
        aa = ((valid[c1] == "A") & (valid[c2] == "A")).sum()
        ab = ((valid[c1] == "A") & (valid[c2] == "B")).sum()
        ba = ((valid[c1] == "B") & (valid[c2] == "A")).sum()
        bb = ((valid[c1] == "B") & (valid[c2] == "B")).sum()
        rows.append({
            "pair": f"{m1}×{m2}",
            "agree": (aa + bb) / n,
            "m1_A_m2_A": aa / n,
            "m1_A_m2_B": ab / n,
            "m1_B_m2_A": ba / n,
            "m1_B_m2_B": bb / n,
            "n": n,
        })
    return pd.DataFrame(rows).set_index("pair")


# ---------------------------------------------------------------------------
# Fleiss' Kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(df: pd.DataFrame) -> float:
    """Compute Fleiss' Kappa for three raters (llama, claude, gpt4) with two
    categories (A, B).

    Only rows where all three models produced a non-null decision are used.
    """
    cols = [f"decision_{m}" for m in _MODELS]
    sub = df[cols].dropna()
    n = len(sub)
    if n == 0:
        return float("nan")

    k = 2   # categories (A, B)
    r = 3   # raters

    # Build ratings matrix: shape (n, k)
    # Column 0: count of 'A' ratings for row i; Column 1: count of 'B' ratings
    ratings = np.zeros((n, k), dtype=float)
    for j, col in enumerate(cols):
        ratings[:, 0] += (sub[col].values == "A").astype(float)
        ratings[:, 1] += (sub[col].values == "B").astype(float)

    # Per-subject agreement
    P_i = (ratings ** 2).sum(axis=1) - r
    P_i = P_i / (r * (r - 1))
    P_bar = P_i.mean()

    # Marginal proportions
    p_j = ratings.sum(axis=0) / (n * r)
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ---------------------------------------------------------------------------
# Game-recognition analysis (Appendix E)
# ---------------------------------------------------------------------------

async def classify_game_recognition(
    df: pd.DataFrame,
    client,  # LLMClient
    justification_col: str = "response_claude",
) -> pd.Series:
    """Use Claude to detect whether each justification mentions game theory.

    Returns a boolean Series aligned with *df* index.
    Named ``game_theory_mentioned``.

    Parameters
    ----------
    client : LLMClient
        Must have a ``claude`` model configured.
    justification_col : str
        Column in *df* containing the raw justification text to analyse.
    """
    logger.info(
        "Classifying game-theory recognition for %d rows using column %s",
        len(df), justification_col,
    )
    results: List[bool] = []
    for text in df[justification_col]:
        if not text:
            results.append(False)
            continue
        prompt = (
            "Does this text explicitly mention the prisoner's dilemma or game theory?\n"
            "Respond only with <YES> or <NO> followed by the relevant sentence(s).\n"
            f"Here is the text: {text}"
        )
        try:
            response = await client.generate(prompt, model="claude")
            raw = (response.get("claude") or "").strip().upper()
            results.append(raw.startswith("<YES>") or raw.startswith("YES"))
        except Exception as e:
            logger.warning("Error classifying justification: %s", e)
            results.append(False)

    return pd.Series(results, index=df.index, name="game_theory_mentioned", dtype=bool)


def cooperation_by_recognition(
    df: pd.DataFrame,
    game_theory_col: str = "game_theory_mentioned",
) -> pd.DataFrame:
    """Cooperation proportion split by game-theory recognition, per model.

    Returns a DataFrame with columns:
    ``model``, ``recognized``, ``cooperation``, ``n``.
    """
    rows = []
    for model in _MODELS:
        dcol = f"decision_{model}"
        for recognized in (True, False):
            sub = df[df[game_theory_col] == recognized]
            n = sub[dcol].notna().sum()
            coop = (sub[dcol] == "A").sum() / n if n > 0 else float("nan")
            rows.append({
                "model": model,
                "recognized": recognized,
                "cooperation": coop,
                "n": n,
            })
    return pd.DataFrame(rows)
