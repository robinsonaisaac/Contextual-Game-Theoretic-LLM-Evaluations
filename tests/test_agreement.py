# tests/test_agreement.py
"""Tests for game_theory_llm.analysis.agreement."""

import pandas as pd
import pytest

from game_theory_llm.analysis.agreement import (
    agreement_by_context,
    all_model_agreement,
    cooperation_by_recognition,
    fleiss_kappa,
    pairwise_agreement,
)


@pytest.fixture
def agree_df():
    """DataFrame where all 3 models always agree (all choose A)."""
    return pd.DataFrame({
        "topic": ["mv_pharma_pro"] * 6,
        "actor_type": ["allies"] * 6,
        "observability": ["private"] * 6,
        "power_dynamic": ["symmetric"] * 6,
        "game_type": ["prisoners_dilemma"] * 6,
        "conversation_mode": ["single_turn"] * 6,
        "decision_llama": ["A"] * 6,
        "decision_claude": ["A"] * 6,
        "decision_gpt4": ["A"] * 6,
    })


@pytest.fixture
def disagree_df():
    """DataFrame where models always disagree (A, B, A per row)."""
    return pd.DataFrame({
        "topic": ["mv_pharma_pro"] * 4,
        "actor_type": ["allies"] * 4,
        "observability": ["private"] * 4,
        "power_dynamic": ["symmetric"] * 4,
        "game_type": ["prisoners_dilemma"] * 4,
        "conversation_mode": ["single_turn"] * 4,
        "decision_llama": ["A", "B", "A", "B"],
        "decision_claude": ["B", "A", "B", "A"],
        "decision_gpt4": ["A", "B", "A", "B"],
    })


class TestAllModelAgreement:
    def test_perfect_agreement(self, agree_df):
        assert all_model_agreement(agree_df) == pytest.approx(1.0)

    def test_zero_all_three_agreement(self, disagree_df):
        # llama and gpt4 always match, but claude always differs → never all-3
        assert all_model_agreement(disagree_df) == pytest.approx(0.0)

    def test_partial_agreement(self):
        df = pd.DataFrame({
            "decision_llama": ["A", "A", "B"],
            "decision_claude": ["A", "B", "B"],
            "decision_gpt4": ["A", "B", "B"],
        })
        # Row 0: all A → agree; Row 1: A,B,B → disagree; Row 2: all B → agree
        assert all_model_agreement(df) == pytest.approx(2 / 3)


class TestAgreementByContext:
    def test_returns_dataframe(self, agree_df):
        result = agreement_by_context(agree_df)
        assert isinstance(result, pd.DataFrame)
        assert "agreement" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns

    def test_perfect_agreement_value(self, agree_df):
        result = agreement_by_context(agree_df)
        assert result["agreement"].iloc[0] == pytest.approx(1.0)


class TestPairwiseAgreement:
    def test_returns_three_pairs(self, agree_df):
        result = pairwise_agreement(agree_df)
        assert len(result) == 3

    def test_perfect_agreement_pairs(self, agree_df):
        result = pairwise_agreement(agree_df)
        assert all(abs(v - 1.0) < 1e-6 for v in result["agree"])

    def test_topic_filter(self, agree_df):
        result = pairwise_agreement(agree_df, topic="mv_pharma_pro")
        assert len(result) == 3

    def test_no_matching_rows(self, agree_df):
        result = pairwise_agreement(agree_df, topic="nonexistent_topic")
        assert result["n"].sum() == 0

    def test_game_type_filter(self, agree_df):
        result = pairwise_agreement(agree_df, game_type="prisoners_dilemma")
        assert len(result) == 3
        assert all(abs(v - 1.0) < 1e-6 for v in result["agree"])

    def test_game_type_filter_no_match(self, agree_df):
        result = pairwise_agreement(agree_df, game_type="stag_hunt")
        assert result["n"].sum() == 0

    def test_conversation_mode_filter(self, agree_df):
        result = pairwise_agreement(agree_df, conversation_mode="single_turn")
        assert len(result) == 3
        assert all(abs(v - 1.0) < 1e-6 for v in result["agree"])

    def test_conversation_mode_filter_no_match(self, agree_df):
        result = pairwise_agreement(agree_df, conversation_mode="multi_turn")
        assert result["n"].sum() == 0


class TestFleissKappa:
    def test_perfect_agreement_is_one(self, agree_df):
        k = fleiss_kappa(agree_df)
        assert k == pytest.approx(1.0)

    def test_returns_float(self, disagree_df):
        k = fleiss_kappa(disagree_df)
        assert isinstance(k, float)

    def test_kappa_in_minus_one_to_one(self, disagree_df):
        k = fleiss_kappa(disagree_df)
        assert -1.0 <= k <= 1.0


class TestCooperationByRecognition:
    def test_structure(self):
        df = pd.DataFrame({
            "decision_llama": ["A", "B", "A", "B"],
            "decision_claude": ["A", "A", "B", "B"],
            "decision_gpt4": ["B", "B", "A", "A"],
            "game_theory_mentioned": [True, True, False, False],
        })
        result = cooperation_by_recognition(df)
        assert "model" in result.columns
        assert "recognized" in result.columns
        assert "cooperation" in result.columns
        assert len(result) == 6  # 3 models × 2 recognized values
