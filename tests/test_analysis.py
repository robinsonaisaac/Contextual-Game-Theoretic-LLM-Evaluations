# tests/test_analysis.py
"""Tests for game_theory_llm.analysis.base."""

import pytest

from game_theory_llm.analysis.base import StoryAnalyzer
from game_theory_llm.models import PayoffMatrix


class TestCalculateExpectedUtility:
    def test_all_choose_a(self):
        m = PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)])
        eu = StoryAnalyzer.calculate_expected_utility(1.0, m)
        assert eu == pytest.approx(3.0)

    def test_all_choose_b(self):
        m = PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)])
        eu = StoryAnalyzer.calculate_expected_utility(0.0, m)
        assert eu == pytest.approx(1.0)

    def test_equal_mix(self):
        m = PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)])
        eu = StoryAnalyzer.calculate_expected_utility(0.5, m)
        # 0.25*3 + 0.25*0 + 0.25*5 + 0.25*1 = 2.25
        assert eu == pytest.approx(2.25)


class TestAnalyzeStories:
    @pytest.mark.asyncio
    async def test_returns_analysis_result(self, mock_client, sample_stories, sample_matrix):
        analyzer = StoryAnalyzer(mock_client)
        result = await analyzer.analyze_stories(sample_stories, sample_matrix)
        assert result.stories == sample_stories
        assert "llama" in result.proportions
        assert "claude" in result.proportions
        assert "gpt4" in result.proportions

    @pytest.mark.asyncio
    async def test_summaries_populated(self, mock_client, sample_stories, sample_matrix):
        analyzer = StoryAnalyzer(mock_client)
        result = await analyzer.analyze_stories(sample_stories, sample_matrix)
        for model in ["llama", "claude", "gpt4"]:
            assert len(result.summaries[model]) > 0

    @pytest.mark.asyncio
    async def test_decisions_extracted(self, mock_client, sample_stories):
        analyzer = StoryAnalyzer(mock_client)
        result = await analyzer.analyze_stories(sample_stories)
        # mock_client returns A for llama and gpt4, B for claude
        assert all(d in ("A", "B", None) for d in result.decisions["llama"])
