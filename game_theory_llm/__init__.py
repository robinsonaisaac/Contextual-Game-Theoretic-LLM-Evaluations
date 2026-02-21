# game_theory_llm/__init__.py
from .models import PayoffMatrix, Story, AnalysisResult, BatchGenerationResult
from .client import LLMClient, ModelConfig, DEFAULT_MODELS, MODEL_REGISTRY, get_models
from .config import (
    ExperimentConfig, get_config, PRESETS,
    Topic, TOPICS, ALL_TOPIC_IDS, HOLDOUT_TOPIC_IDS,
    OBSERVABILITY, POWER_DYNAMIC,
    get_topics_by_axis, get_matched_pairs,
)
from .generator import StoryGenerator
from .decision_parser import extract_decision
from .analysis import (
    StoryAnalyzer,
    StatisticalAnalyzer,
    all_model_agreement,
    agreement_by_context,
    pairwise_agreement,
    fleiss_kappa,
    classify_game_recognition,
    cooperation_by_recognition,
    CategoricalPredictor,
    EmbeddingPredictor,
    evaluate_all_models,
    cooperation_by_model,
)
from .analysis.visualization import (
    plot_decision_distributions,
    plot_enhanced_visualizations,
    plot_cooperation_heatmaps,
    plot_agreement_by_context,
    plot_pairwise_agreement,
    plot_swap_distribution,
    plot_swap_delta_heatmaps,
    plot_defection_by_model,
    plot_mmlu_vs_defection,
    plot_cramers_v_by_model,
    plot_game_recognition,
)

__all__ = [
    # models
    "PayoffMatrix", "Story", "AnalysisResult", "BatchGenerationResult",
    # client
    "LLMClient", "ModelConfig", "DEFAULT_MODELS", "MODEL_REGISTRY", "get_models",
    # config
    "ExperimentConfig", "get_config", "PRESETS",
    "Topic", "TOPICS", "ALL_TOPIC_IDS", "HOLDOUT_TOPIC_IDS",
    "OBSERVABILITY", "POWER_DYNAMIC",
    "get_topics_by_axis", "get_matched_pairs",
    # generation
    "StoryGenerator",
    # parsing
    "extract_decision",
    # analysis
    "StoryAnalyzer", "StatisticalAnalyzer",
    # agreement
    "all_model_agreement", "agreement_by_context", "pairwise_agreement",
    "fleiss_kappa", "classify_game_recognition", "cooperation_by_recognition",
    # predictive
    "CategoricalPredictor", "EmbeddingPredictor",
    "evaluate_all_models", "cooperation_by_model",
    # visualization
    "plot_decision_distributions", "plot_enhanced_visualizations",
    "plot_cooperation_heatmaps", "plot_agreement_by_context",
    "plot_pairwise_agreement", "plot_swap_distribution",
    "plot_swap_delta_heatmaps", "plot_defection_by_model",
    "plot_mmlu_vs_defection", "plot_cramers_v_by_model",
    "plot_game_recognition",
]
