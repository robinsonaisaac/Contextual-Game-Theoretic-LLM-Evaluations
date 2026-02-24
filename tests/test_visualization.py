# tests/test_visualization.py
"""Smoke tests for all paper-figure visualization functions."""

import os
import tempfile

import pandas as pd
import pytest

from game_theory_llm.analysis.visualization import (
    plot_agreement_by_context,
    plot_cooperation_heatmaps,
    plot_cramers_v_by_model,
    plot_defection_by_model,
    plot_focal_rate_by_game,
    plot_game_recognition,
    plot_mmlu_vs_defection,
    plot_pairwise_agreement,
    plot_single_vs_multi_turn,
    plot_swap_delta_heatmaps,
    plot_swap_distribution,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame with original + swapped decisions."""
    return pd.DataFrame({
        "topic": ["mv_pharma_pro", "mv_pharma_pro", "pol_dem_rep", "pol_dem_rep"],
        "actor_type": ["allies", "enemies", "allies", "enemies"],
        "observability": ["private", "public", "private", "public"],
        "power_dynamic": ["symmetric", "asymmetric", "symmetric", "asymmetric"],
        "game_type": ["prisoners_dilemma", "prisoners_dilemma", "stag_hunt", "stag_hunt"],
        "conversation_mode": ["single_turn", "single_turn", "single_turn", "single_turn"],
        "decision_llama": ["A", "B", "A", "A"],
        "decision_claude": ["A", "A", "B", "B"],
        "decision_gpt4": ["B", "B", "A", "A"],
        "decision_swapped_llama": ["B", "A", "B", "B"],
        "decision_swapped_claude": ["B", "B", "A", "A"],
        "decision_swapped_gpt4": ["A", "A", "B", "B"],
    })


@pytest.fixture
def agreement_df():
    return pd.DataFrame({
        "topic": ["mv_pharma_pro", "pol_dem_rep"],
        "actor_type": ["allies", "enemies"],
        "agreement": [0.8, 0.4],
        "ci_lower": [0.7, 0.3],
        "ci_upper": [0.9, 0.5],
        "n": [50, 50],
    })


@pytest.fixture
def pairwise_df():
    import pandas as pd
    data = {
        "pair": ["llama×claude", "llama×gpt4", "claude×gpt4"],
        "agree": [0.75, 0.60, 0.65],
        "m1_A_m2_A": [0.40, 0.30, 0.35],
        "m1_A_m2_B": [0.10, 0.20, 0.15],
        "m1_B_m2_A": [0.15, 0.20, 0.20],
        "m1_B_m2_B": [0.35, 0.30, 0.30],
        "n": [100, 100, 100],
    }
    return pd.DataFrame(data).set_index("pair")


def test_plot_cooperation_heatmaps(sample_df):
    with tempfile.TemporaryDirectory() as d:
        plot_cooperation_heatmaps(sample_df, save_dir=d)
        files = os.listdir(d)
        assert any("fig4_cooperation_heatmap" in f for f in files)


def test_plot_agreement_by_context(agreement_df):
    with tempfile.TemporaryDirectory() as d:
        plot_agreement_by_context(agreement_df, save_dir=d)
        assert "fig5_agreement_by_context.png" in os.listdir(d)


def test_plot_pairwise_agreement(pairwise_df):
    with tempfile.TemporaryDirectory() as d:
        plot_pairwise_agreement(pairwise_df, context_label="test", save_dir=d)
        assert any("fig6" in f for f in os.listdir(d))


def test_plot_swap_distribution(sample_df):
    with tempfile.TemporaryDirectory() as d:
        plot_swap_distribution(sample_df, save_dir=d)
        assert "fig7_swap_distribution.png" in os.listdir(d)


def test_plot_swap_delta_heatmaps(sample_df):
    with tempfile.TemporaryDirectory() as d:
        plot_swap_delta_heatmaps(sample_df, save_dir=d)
        files = os.listdir(d)
        assert any("fig8_swap_delta" in f for f in files)


def test_plot_defection_by_model():
    model_df = pd.DataFrame({
        "model": ["Llama", "Claude", "GPT-4o"],
        "defection_rate": [0.53, 0.48, 0.41],
        "cooperation_rate": [0.47, 0.52, 0.59],
    })
    with tempfile.TemporaryDirectory() as d:
        plot_defection_by_model(model_df, save_dir=d)
        assert "fig9a_defection_by_model.png" in os.listdir(d)


def test_plot_mmlu_vs_defection():
    model_df = pd.DataFrame({
        "model": ["ModelA", "ModelB", "ModelC"],
        "mmlu_score": [80.0, 75.0, 70.0],
        "defection_rate": [0.6, 0.5, 0.4],
        "family": ["Llama", "Llama", "Qwen"],
    })
    with tempfile.TemporaryDirectory() as d:
        plot_mmlu_vs_defection(model_df, save_dir=d)
        assert "fig9b_mmlu_vs_defection.png" in os.listdir(d)


def test_plot_cramers_v_by_model(sample_df):
    with tempfile.TemporaryDirectory() as d:
        plot_cramers_v_by_model(sample_df, save_dir=d)
        assert "fig9c_cramers_v_by_model.png" in os.listdir(d)


def test_plot_focal_rate_by_game(sample_df):
    with tempfile.TemporaryDirectory() as d:
        plot_focal_rate_by_game(sample_df, save_dir=d)
        assert "focal_rate_by_game.png" in os.listdir(d)


def test_plot_single_vs_multi_turn():
    df = pd.DataFrame({
        "topic": ["mv_pharma_pro"] * 4,
        "actor_type": ["allies"] * 4,
        "conversation_mode": ["single_turn", "single_turn", "multi_turn", "multi_turn"],
        "decision_llama": ["A", "B", "A", "A"],
        "decision_claude": ["A", "A", "B", "B"],
        "decision_gpt4": ["B", "B", "A", "A"],
    })
    with tempfile.TemporaryDirectory() as d:
        plot_single_vs_multi_turn(df, save_dir=d)
        assert "single_vs_multi_turn.png" in os.listdir(d)


def test_plot_single_vs_multi_turn_skips_single_mode(sample_df):
    """Should not produce a plot when only one conversation mode exists."""
    with tempfile.TemporaryDirectory() as d:
        plot_single_vs_multi_turn(sample_df, save_dir=d)
        assert "single_vs_multi_turn.png" not in os.listdir(d)


def test_plot_game_recognition():
    rec_df = pd.DataFrame({
        "model": ["llama", "llama", "claude", "claude", "gpt4", "gpt4"],
        "recognized": [True, False, True, False, True, False],
        "cooperation": [0.7, 0.5, 0.6, 0.55, 0.65, 0.52],
        "n": [30, 70, 40, 60, 35, 65],
    })
    with tempfile.TemporaryDirectory() as d:
        plot_game_recognition(rec_df, save_dir=d)
        assert "fig10_game_recognition.png" in os.listdir(d)
