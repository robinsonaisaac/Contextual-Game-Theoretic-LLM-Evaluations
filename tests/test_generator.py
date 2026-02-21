# tests/test_generator.py
"""Tests for game_theory_llm.generator."""

import pytest

from game_theory_llm.generator import StoryGenerator
from game_theory_llm.models import PayoffMatrix


class TestCreateQuery:
    def test_produces_string(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "business", "real_world", "allies",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_topic(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "international business", "real_world", "allies",
        )
        assert "international business" in prompt

    def test_contains_decision_labels(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "business", "real_world", "allies",
        )
        assert "Decision A" in prompt or "decision" in prompt.lower()

    def test_contains_story_tags(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "business", "real_world", "allies",
        )
        assert "<story>" in prompt

    def test_typo_fixed_circumstance(self, mock_client, sample_matrix):
        """Bug #6: 'cirumstance' should be 'circumstance'."""
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "business", "real_world", "allies",
        )
        assert "cirumstance" not in prompt
        assert "circumstance" in prompt


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_extracts_stories(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "business", "real_world", "allies",
        )
        assert len(result.stories) == 2

    @pytest.mark.asyncio
    async def test_extracts_decisions(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "business", "real_world", "allies",
        )
        assert result.stories[0].decision == "A"
        assert result.stories[1].decision == "B"


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_rejects_unknown_topic(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid topic"):
            await gen.generate_stories(
                sample_matrix, "underwater basket weaving",
                "real_world", "allies", n_stories=1,
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_world(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid world type"):
            await gen.generate_stories(
                sample_matrix, "business", "mars", "allies", n_stories=1,
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_actor(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid actor type"):
            await gen.generate_stories(
                sample_matrix, "business", "real_world", "frenemies", n_stories=1,
            )
