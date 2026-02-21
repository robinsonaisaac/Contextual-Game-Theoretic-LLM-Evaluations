# game_theory_llm/analysis/statistical.py
"""Enhanced statistical analysis — chi-square, Fisher, Cramer's V, runs test, entropy."""

from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.sandbox.stats.runs import runstest_1samp

from .._logging import get_logger
from ..models import AnalysisResult, PayoffMatrix, Story
from .base import StoryAnalyzer

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Wraps a :class:`StoryAnalyzer` and adds statistical tests via composition.

    Parameters
    ----------
    base_analyzer : StoryAnalyzer
        The base analyser used for decision extraction and proportion
        calculation.
    """

    def __init__(self, base_analyzer: StoryAnalyzer):
        self.base = base_analyzer

    async def analyze_stories(
        self,
        stories: List[Story],
        payoff_matrix: Optional[PayoffMatrix] = None,
        df_existing: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> AnalysisResult:
        """Run base analysis then enhance with statistical tests."""
        base_results = await self.base.analyze_stories(
            stories, payoff_matrix, df_existing, progress_callback,
        )
        return self._enhance(stories, base_results, payoff_matrix, df_existing)

    # ------------------------------------------------------------------

    def _enhance(
        self,
        stories: List[Story],
        base_results: AnalysisResult,
        payoff_matrix: Optional[PayoffMatrix],
        df_existing: Optional[pd.DataFrame] = None,
    ) -> AnalysisResult:
        """Add chi-square, Fisher, Cramer's V, runs test, entropy."""
        logger.info("Enhancing analysis with statistical measures")

        if df_existing is not None:
            df = df_existing
        else:
            df = pd.DataFrame({
                "topic": [s.topic for s in stories],
                "world_type": [s.world_type for s in stories],
                "actor_type": [s.actor_type for s in stories],
            })
            for model in ["llama", "claude", "gpt4"]:
                df[f"decision_{model}"] = base_results.decisions[model]

        enhanced_summaries: Dict[str, List[str]] = {m: [] for m in ["llama", "claude", "gpt4"]}

        for model in ["llama", "claude", "gpt4"]:
            col = f"decision_{model}"

            # 1. Chi-square + Fisher's exact
            for category in ("topic", "world_type", "actor_type"):
                contingency = pd.crosstab(df[category], df[col])
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                enhanced_summaries[model].append(
                    f"chi2_independence_{category}: p={p_value:.6f}, chi2={chi2:.2f}"
                )
                if contingency.shape == (2, 2):
                    odds_ratio, fisher_p = fisher_exact(contingency)
                    enhanced_summaries[model].append(
                        f"fisher_exact_{category}: p={fisher_p:.6f}, odds_ratio={odds_ratio:.2f}"
                    )

            # 2. Multiple testing correction
            p_values = [
                float(s.split("p=")[1].split(",")[0])
                for s in enhanced_summaries[model]
                if "p=" in s
            ]
            if p_values:
                rejected, corrected_p, _, _ = multipletests(p_values, method="bonferroni")
                enhanced_summaries[model].append(
                    f"multiple_testing_correction: {list(zip(rejected, corrected_p))}"
                )

            # 3. Cramer's V
            for category in ("topic", "world_type", "actor_type"):
                contingency = pd.crosstab(df[category], df[col])
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                if n > 0 and min_dim > 0:
                    chi2, _ = stats.chi2_contingency(contingency)[:2]
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    enhanced_summaries[model].append(f"cramers_v_{category}: {cramers_v:.4f}")

            # 4. Interaction analysis
            for cat1, cat2 in combinations(("topic", "world_type", "actor_type"), 2):
                observed = pd.crosstab(index=df[cat1], columns=df[cat2])
                chi2, p_value, _, _ = chi2_contingency(observed)
                enhanced_summaries[model].append(f"interaction_{cat1}_{cat2}: p={p_value:.6f}")

            # 5. Runs test
            decisions_numeric = [1 if d == "A" else 0 for d in df[col].values]
            zstat, rt_p = runstest_1samp(decisions_numeric, cutoff="mean")
            enhanced_summaries[model].append(
                f"decision_pattern_stability: zstat={zstat:.3f}, p={rt_p:.6f}"
            )

            # 6. Conditional entropy
            for category in ("topic", "world_type", "actor_type"):
                joint_prob = pd.crosstab(df[category], df[col], normalize="all")
                category_prob = joint_prob.sum(axis=1)
                cond_entropy = 0.0
                for i in joint_prob.index:
                    for j in joint_prob.columns:
                        if joint_prob.loc[i, j] > 0:
                            cond_entropy -= joint_prob.loc[i, j] * np.log2(
                                joint_prob.loc[i, j] / category_prob[i]
                            )
                enhanced_summaries[model].append(f"conditional_entropy_{category}: {cond_entropy:.4f}")

        return AnalysisResult(
            stories=base_results.stories,
            decisions=base_results.decisions,
            summaries=enhanced_summaries,
            proportions=base_results.proportions,
            by_topic=base_results.by_topic,
            by_world=base_results.by_world,
            by_actor=base_results.by_actor,
        )
