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


class TestCrossGameFocalRateTable:
    def _make_stories(self):
        from game_theory_llm.models import Story
        return [
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="prisoners_dilemma", decision="A"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="prisoners_dilemma", decision="B"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="A"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="B"),
            Story(content="x", topic="t1", actor_type="allies",
                  game_type="deadlock", decision="B"),
        ]

    def test_returns_dataframe_with_expected_columns(self):
        from game_theory_llm.analysis.base import cross_game_focal_rate_table
        df = cross_game_focal_rate_table(self._make_stories())
        assert "game_type" in df.columns
        assert "focal_a_rate" in df.columns

    def test_focal_rate_values(self):
        from game_theory_llm.analysis.base import cross_game_focal_rate_table
        df = cross_game_focal_rate_table(self._make_stories())
        pd_row = df[df["game_type"] == "prisoners_dilemma"].iloc[0]
        dl_row = df[df["game_type"] == "deadlock"].iloc[0]
        assert pd_row["focal_a_rate"] == 0.5
        assert dl_row["focal_a_rate"] == pytest.approx(1/3)
