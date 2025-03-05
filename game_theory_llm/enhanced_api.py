# game_theory_llm/enhanced_api.py
from typing import List, Dict, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as nx
from itertools import combinations, product
import logging
from dataclasses import dataclass
import os
from .api import APIClient, Story, PayoffMatrix, AnalysisResult
from statsmodels.sandbox.stats.runs import runstest_1samp

logger = logging.getLogger(__name__)

class EnhancedAPIClient(APIClient):
    """Enhanced API client with advanced analysis while maintaining original functionality."""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "analysis_results"):
        super().__init__(api_key)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def analyze_stories(
        self,
        stories: List[Story],
        payoff_matrix: Optional[PayoffMatrix] = None,
        df_existing=None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> AnalysisResult:
        """Override of original analyze_stories with enhanced capabilities."""
        if df_existing is not None:
            base_results = await super().analyze_stories(stories, payoff_matrix, df_existing, progress_callback)
            enhanced_results = await self._enhance_analysis(stories, base_results, payoff_matrix, df_existing)
        else:
            base_results = await super().analyze_stories(stories, payoff_matrix, progress_callback=progress_callback)
            enhanced_results = await self._enhance_analysis(stories, base_results, payoff_matrix)
        return enhanced_results
    
    async def _enhance_analysis(
        self,
        stories: List[Story],
        base_results: AnalysisResult,
        payoff_matrix: Optional[PayoffMatrix],
        df_existing=None
    ) -> AnalysisResult:
        """Enhance analysis with additional statistical measures."""
        logger.info("Enhancing analysis with statistical measures")
        
        # If we have an existing DataFrame, use it; otherwise create from stories
        if df_existing is not None:
            df = df_existing
        else:
            df = pd.DataFrame({
                'topic': [s.topic for s in stories],
                'world_type': [s.world_type for s in stories],
                'actor_type': [s.actor_type for s in stories]
            })
            # Add decision columns for each model
            for model in ['llama', 'claude', 'gpt4']:
                df[f'decision_{model}'] = base_results.decisions[model]

        # Summaries for each model stored in a dict
        enhanced_summaries = {model: [] for model in ['llama', 'claude', 'gpt4']}
        
        # Perform statistical tests for each model
        for model in ['llama', 'claude', 'gpt4']:
            decision_col = f'decision_{model}'
            
            # 1. Chi-square tests of independence + Fisher's exact test for 2x2
            for category in ['topic', 'world_type', 'actor_type']:
                contingency = pd.crosstab(df[category], df[decision_col])
                
                # Chi-square test
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                enhanced_summaries[model].append(f"chi2_independence_{category}: p={p_value:.6f}, chi2={chi2:.2f}")
                
                # Fisher's exact test for 2x2 tables
                if contingency.shape == (2, 2):
                    odds_ratio, fisher_p = fisher_exact(contingency)
                    enhanced_summaries[model].append(
                        f"fisher_exact_{category}: p={fisher_p:.6f}, odds_ratio={odds_ratio:.2f}"
                    )
            
            # 2. Multiple testing correction
            p_values = [
                float(s.split('p=')[1].split(',')[0]) 
                for s in enhanced_summaries[model] 
                if 'p=' in s
            ]
            if p_values:
                rejected, corrected_p, _, _ = multipletests(p_values, method='bonferroni')
                enhanced_summaries[model].append(
                    f"multiple_testing_correction: {list(zip(rejected, corrected_p))}"
                )
            
            # 3. Effect sizes (Cramer's V)
            for category in ['topic', 'world_type', 'actor_type']:
                contingency = pd.crosstab(df[category], df[decision_col])
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                if n > 0 and min_dim > 0:
                    chi2, _ = stats.chi2_contingency(contingency)[:2]
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    enhanced_summaries[model].append(f"cramers_v_{category}: {cramers_v:.4f}")
            
            # 4. Interaction analysis between pairs of variables
            for cat1, cat2 in combinations(['topic', 'world_type', 'actor_type'], 2):
                # Create two-way contingency table for the categorical variables
                observed = pd.crosstab(index=df[cat1], columns=df[cat2])
                
                # Perform chi-square test on the contingency table
                chi2, p_value, _, _ = chi2_contingency(observed)
                enhanced_summaries[model].append(f"interaction_{cat1}_{cat2}: p={p_value:.6f}")
            
            # 5. Decision pattern stability (Runs test)
            decisions = df[decision_col].values
            # Convert A/B to numeric (1/0)
            decisions_numeric = [1 if d == 'A' else 0 for d in decisions]
            
            # statsmodels' runstest_1samp returns z-statistic, p-value
            zstat, rt_p_value = runstest_1samp(decisions_numeric, cutoff='mean')
            enhanced_summaries[model].append(
                f"decision_pattern_stability: zstat={zstat:.3f}, p={rt_p_value:.6f}"
            )
            
            # 6. Conditional entropy to measure predictability
            for category in ['topic', 'world_type', 'actor_type']:
                joint_prob = pd.crosstab(df[category], df[decision_col], normalize='all')
                category_prob = joint_prob.sum(axis=1)
                
                cond_entropy = 0
                for i in joint_prob.index:
                    for j in joint_prob.columns:
                        if joint_prob.loc[i, j] > 0:
                            cond_entropy -= joint_prob.loc[i, j] * np.log2(
                                joint_prob.loc[i, j] / category_prob[i]
                            )
                
                enhanced_summaries[model].append(
                    f"conditional_entropy_{category}: {cond_entropy:.4f}"
                )

        # Return a new AnalysisResult containing the original base results plus new summaries
        return AnalysisResult(
            stories=base_results.stories,
            decisions=base_results.decisions,
            summaries=enhanced_summaries,  # dictionary of model -> list of strings
            proportions=base_results.proportions,
            by_topic=base_results.by_topic,
            by_world=base_results.by_world,
            by_actor=base_results.by_actor
        )
    
    def _generate_enhanced_visualizations(
        self,
        df: pd.DataFrame,
        statistical_tests: Dict[str, Dict[str, float]],
        decision_patterns: Dict[str, Dict],
        interaction_effects: Dict[str, float],
        clustering_results: Dict[str, Any],
        payoff_matrix: Optional[PayoffMatrix]
    ):
        """Generate extended visualizations for each model separately."""
        logger.info("Generating enhanced visualizations for each model")
        
        for model in ['llama', 'claude', 'gpt4']:
            model_dir = f"{self.output_dir}/{model}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a copy of the DataFrame with this model's decisions
            model_df = df.copy()
            
            # Add both normal and swapped decisions
            model_df['decision'] = df[f'decision_{model}']
            model_df['decision_swapped'] = df[f'decision_swapped_{model}']
            model_df['response'] = df[f'response_{model}']
            model_df['response_swapped'] = df[f'response_swapped_{model}']
            
            # Expected Value Analysis (if payoff matrix provided)
            if payoff_matrix is not None:
                for decision_type in ['', '_swapped']:
                    prob_a_range = np.linspace(0, 1, 100)
                    def _calc_ev(p):
                        p_b = 1 - p
                        return (
                            p * p * payoff_matrix.matrix[0][0] +
                            p * p_b * payoff_matrix.matrix[1][0] +
                            p_b * p * payoff_matrix.matrix[2][0] +
                            p_b * p_b * payoff_matrix.matrix[3][0]
                        )
                    
                    ev_values = [_calc_ev(pr) for pr in prob_a_range]
                    plt.figure(figsize=(8, 5))
                    plt.plot(prob_a_range, ev_values)
                    suffix = " (Swapped)" if decision_type else ""
                    plt.title(f"Expected Utility vs. Probability of A - {model.upper()}{suffix}")
                    plt.xlabel("Probability of choosing A")
                    plt.ylabel("Expected Utility")
                    plt.savefig(f"{model_dir}/expected_value_analysis{decision_type}.png")
                    plt.close()

            # Decision distribution by category (both normal and swapped)
            for category in ['topic', 'world_type', 'actor_type']:
                for decision_type in ['decision', 'decision_swapped']:
                    plt.figure(figsize=(10, 6))
                    contingency = pd.crosstab(model_df[category], model_df[decision_type])
                    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                    contingency_pct.plot(kind='bar', stacked=True)
                    suffix = " (Swapped)" if decision_type.endswith('_swapped') else ""
                    plt.title(f"Decision Distribution by {category} - {model.upper()}{suffix}")
                    plt.xlabel(category)
                    plt.ylabel("Percentage")
                    plt.legend(title="Decision")
                    plt.tight_layout()
                    plt.savefig(f"{model_dir}/decision_by_{category}_{decision_type}.png")
                    plt.close()

            # Comparison plots (Normal vs Swapped)
            for category in ['topic', 'world_type', 'actor_type']:
                plt.figure(figsize=(12, 6))
                
                # Create side-by-side comparison
                plt.subplot(1, 2, 1)
                contingency = pd.crosstab(model_df[category], model_df['decision'])
                contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                contingency_pct.plot(kind='bar', stacked=True)
                plt.title("Normal")
                plt.xlabel(category)
                plt.ylabel("Percentage")
                
                plt.subplot(1, 2, 2)
                contingency = pd.crosstab(model_df[category], model_df['decision_swapped'])
                contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                contingency_pct.plot(kind='bar', stacked=True)
                plt.title("Swapped")
                plt.xlabel(category)
                
                plt.suptitle(f"Decision Distribution Comparison by {category} - {model.upper()}")
                plt.tight_layout()
                plt.savefig(f"{model_dir}/decision_comparison_{category}.png")
                plt.close()

            # Heatmaps for both normal and swapped decisions
            for decision_type in ['decision', 'decision_swapped']:
                plt.figure(figsize=(12, 8))
                pivot_table = pd.crosstab(
                    [model_df['topic'], model_df['world_type']], 
                    [model_df['actor_type'], model_df[decision_type]]
                )
                sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
                suffix = " (Swapped)" if decision_type.endswith('_swapped') else ""
                plt.title(f"Decision Heatmap - {model.upper()}{suffix}")
                plt.tight_layout()
                plt.savefig(f"{model_dir}/decision_heatmap_{decision_type}.png")
                plt.close()

            # Decision Change Analysis (comparing normal vs swapped)
            plt.figure(figsize=(8, 6))
            decision_changes = (model_df['decision'] != model_df['decision_swapped']).value_counts()
            decision_changes.plot(kind='bar')
            plt.title(f"Decision Changes After Swapping - {model.upper()}")
            plt.xlabel("Decision Changed")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{model_dir}/decision_changes.png")
            plt.close()

            # Statistical tests visualization
            model_tests = {k: v for k, v in statistical_tests.items() if k.startswith(model)}
            if model_tests:
                plt.figure(figsize=(10, 6))
                test_names = list(model_tests.keys())
                p_values = [test['p_value'] for test in model_tests.values()]
                plt.barh(range(len(p_values)), p_values)
                plt.yticks(range(len(test_names)), [t.replace(f"{model}_", "") for t in test_names])
                plt.axvline(x=0.05, color='r', linestyle='--', label='α=0.05')
                plt.title(f"Statistical Tests P-values - {model.upper()}")
                plt.xlabel("P-value")
                plt.tight_layout()
                plt.savefig(f"{model_dir}/statistical_tests.png")
                plt.close()

            # Consistency Analysis (agreement between normal and swapped decisions)
            plt.figure(figsize=(8, 6))
            consistency_matrix = pd.crosstab(
                model_df['decision'],
                model_df['decision_swapped'],
                normalize='all'
            ) * 100
            sns.heatmap(consistency_matrix, annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title(f"Decision Consistency Matrix - {model.upper()}\n(% of total decisions)")
            plt.xlabel("Swapped Decision")
            plt.ylabel("Original Decision")
            plt.tight_layout()
            plt.savefig(f"{model_dir}/decision_consistency_matrix.png")
            plt.close()

        logger.info("Enhanced visualizations generated for all models")