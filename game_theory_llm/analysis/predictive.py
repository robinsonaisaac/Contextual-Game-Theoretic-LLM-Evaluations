# game_theory_llm/analysis/predictive.py
"""XGBoost predictive models from Section 4 / Appendix C.

Two predictors are implemented:

CategoricalPredictor  (Table 4a)
    Features: one-hot topic + actor_type + world_type + label_order (0/1)
    label_order: 0 = original (A=Cooperate), 1 = swapped (B=Cooperate)

EmbeddingPredictor  (Table 4b)
    Features: 384-dim sentence embeddings via ``all-distilroberta-v1``
    from the Sentence Transformers library.

Both use XGBClassifier with an 80/20 train/test split and grid search
over the parameters detailed in Table 5 of Appendix C.

Usage
-----
    from game_theory_llm.analysis.predictive import evaluate_all_models
    results = evaluate_all_models(df)   # returns DataFrame matching Table 4
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._logging import get_logger

logger = get_logger(__name__)

_MODELS = ("llama", "claude", "gpt4")

# Appendix C grid-search space
_PARAM_GRID = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200, 500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 1.0],
}


def _build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide DataFrame (one row per story) to long format.

    Each story appears twice:
      - label_order=0: original (decision_A = Cooperate)
      - label_order=1: swapped  (decision_B = Cooperate)

    The target column ``cooperate`` is 1 if the model cooperated.
    Returns one row per (story × label_order × model) combination
    but typically callers split by model.
    """
    required = {"topic", "world_type", "actor_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    rows = []
    has_swapped = any(f"decision_swapped_{m}" in df.columns for m in _MODELS)

    for _, row in df.iterrows():
        base = {
            "topic": row["topic"],
            "world_type": row["world_type"],
            "actor_type": row["actor_type"],
            "story_content": row.get("story_content", ""),
        }
        for model in _MODELS:
            # Original pass
            dec = row.get(f"decision_{model}")
            if dec is not None:
                rows.append({
                    **base,
                    "model": model,
                    "label_order": 0,
                    "cooperate": int(dec == "A"),
                })
            # Swapped pass
            if has_swapped:
                dec_sw = row.get(f"decision_swapped_{model}")
                if dec_sw is not None:
                    rows.append({
                        **base,
                        "model": model,
                        "label_order": 1,
                        "cooperate": int(dec_sw == "B"),  # B=Cooperate when swapped
                    })

    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute accuracy, F1, Brier score, AUROC."""
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Categorical predictor
# ---------------------------------------------------------------------------

class CategoricalPredictor:
    """XGBoost classifier on one-hot (topic, actor, world, label_order).

    Reproduces Table 4a.
    """

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._encoders: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalPredictor":
        """Fit one XGBClassifier per LLM with grid-search.

        Parameters
        ----------
        df : DataFrame
            Wide format (output of ``build_dataframe``).
        """
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier

        long = _build_long_df(df)

        for model in _MODELS:
            sub = long[long["model"] == model].dropna(subset=["cooperate"])
            if len(sub) < 10:
                logger.warning("Too few samples for %s, skipping", model)
                continue

            feat_cols = ["topic", "world_type", "actor_type"]
            X_cat = sub[feat_cols]
            X_order = sub[["label_order"]].values

            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_enc = enc.fit_transform(X_cat)
            X = np.hstack([X_enc, X_order])
            y = sub["cooperate"].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = GridSearchCV(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                ),
                _PARAM_GRID,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            clf.fit(X_tr, y_tr)
            self._models[model] = (clf, X_te, y_te)
            self._encoders[model] = enc
            logger.info(
                "CategoricalPredictor(%s) best params: %s", model, clf.best_params_
            )

        return self

    def evaluate(self) -> pd.DataFrame:
        """Return accuracy / F1 / Brier / AUROC for each LLM.

        Returns a DataFrame matching Table 4a.
        """
        rows = []
        for model, (clf, X_te, y_te) in self._models.items():
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
            m = _metrics(y_te, y_pred, y_prob)
            rows.append({"model": model, **m})
        return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Embedding predictor
# ---------------------------------------------------------------------------

class EmbeddingPredictor:
    """XGBoost classifier on sentence-transformer embeddings.

    Reproduces Table 4b.  Uses ``all-distilroberta-v1`` (384-dim) from
    the Sentence Transformers library, matching the paper.
    """

    def __init__(self, embedding_model: str = "all-distilroberta-v1"):
        self.embedding_model_name = embedding_model
        self._embedder = None
        self._models: Dict[str, object] = {}

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def fit(self, df: pd.DataFrame) -> "EmbeddingPredictor":
        """Fit one XGBClassifier per LLM using story embeddings + label_order."""
        from sklearn.model_selection import GridSearchCV, train_test_split
        from xgboost import XGBClassifier

        long = _build_long_df(df)
        embedder = self._get_embedder()

        # Compute embeddings once for unique story contents
        unique_contents = long["story_content"].unique().tolist()
        logger.info("Embedding %d unique stories...", len(unique_contents))
        emb_matrix = embedder.encode(unique_contents, show_progress_bar=True)
        emb_map = {content: emb for content, emb in zip(unique_contents, emb_matrix)}

        for model in _MODELS:
            sub = long[long["model"] == model].dropna(subset=["cooperate"])
            if len(sub) < 10:
                logger.warning("Too few samples for %s, skipping", model)
                continue

            X_emb = np.vstack([emb_map[c] for c in sub["story_content"]])
            X_order = sub[["label_order"]].values
            X = np.hstack([X_emb, X_order])
            y = sub["cooperate"].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = GridSearchCV(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                ),
                _PARAM_GRID,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            clf.fit(X_tr, y_tr)
            self._models[model] = (clf, X_te, y_te)
            logger.info(
                "EmbeddingPredictor(%s) best params: %s", model, clf.best_params_
            )

        return self

    def evaluate(self) -> pd.DataFrame:
        """Return accuracy / F1 / Brier / AUROC for each LLM.

        Returns a DataFrame matching Table 4b.
        """
        rows = []
        for model, (clf, X_te, y_te) in self._models.items():
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
            m = _metrics(y_te, y_pred, y_prob)
            rows.append({"model": model, **m})
        return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Convenience: evaluate both predictors
# ---------------------------------------------------------------------------

def evaluate_all_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit both predictors and return (table_4a, table_4b).

    Parameters
    ----------
    df : DataFrame
        Wide format DataFrame from ``StoryAnalyzer.build_dataframe``.

    Returns
    -------
    (categorical_results, embedding_results) : tuple of DataFrames
        Each matches the corresponding part of Table 4 in the paper.
    """
    logger.info("Training categorical predictor...")
    cat = CategoricalPredictor().fit(df)
    cat_results = cat.evaluate()

    logger.info("Training embedding predictor...")
    emb = EmbeddingPredictor().fit(df)
    emb_results = emb.evaluate()

    return cat_results, emb_results


# ---------------------------------------------------------------------------
# Multi-model MMLU analysis (Appendix D)
# ---------------------------------------------------------------------------

def cooperation_by_model(
    model_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate cooperation rates across many models.

    Parameters
    ----------
    model_results : dict[str, DataFrame]
        Keys are model names; values are DataFrames with a ``decision``
        column (A/B) where A=Cooperate.

    Returns a DataFrame with ``model``, ``defection_rate``, ``cooperation_rate``.
    """
    rows = []
    for model_name, mdf in model_results.items():
        n = mdf["decision"].notna().sum()
        coop = (mdf["decision"] == "A").sum() / n if n > 0 else float("nan")
        rows.append({
            "model": model_name,
            "cooperation_rate": coop,
            "defection_rate": 1 - coop,
        })
    return pd.DataFrame(rows).sort_values("defection_rate", ascending=False).reset_index(drop=True)
