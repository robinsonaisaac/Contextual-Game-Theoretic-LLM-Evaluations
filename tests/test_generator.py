# tests/test_generator.py
"""Tests for game_theory_llm.generator."""

import pytest

from game_theory_llm.generator import StoryGenerator
from game_theory_llm.models import PayoffMatrix


class TestCreateQuery:
    def test_produces_string(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_scenario(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert "pharmaceutical" in prompt

    def test_contains_decision_labels(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert "Decision A" in prompt or "decision" in prompt.lower()

    def test_contains_story_tags(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert "<story>" in prompt

    def test_contains_observability(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
            observability="public",
        )
        assert "public" in prompt.lower()

    def test_contains_power_dynamic(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
            power_dynamic="asymmetric",
        )
        assert "asymmetric" in prompt.lower() or "significantly" in prompt.lower()

    def test_no_circumstance_typo(self, mock_client, sample_matrix):
        """Bug #6: 'cirumstance' should not appear."""
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert "cirumstance" not in prompt


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_extracts_stories(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert len(result.stories) == 2

    @pytest.mark.asyncio
    async def test_extracts_decisions(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert result.stories[0].decision == "A"
        assert result.stories[1].decision == "B"

    @pytest.mark.asyncio
    async def test_stories_carry_dimensions(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "mv_pharma_pro", "enemies",
            observability="public", power_dynamic="asymmetric",
        )
        for story in result.stories:
            assert story.actor_type == "enemies"
            assert story.observability == "public"
            assert story.power_dynamic == "asymmetric"


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_rejects_unknown_topic(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid topic"):
            await gen.generate_stories(
                sample_matrix, "underwater basket weaving",
                "allies", n_stories=1,
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_actor(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid actor type"):
            await gen.generate_stories(
                sample_matrix, "mv_pharma_pro",
                "frenemies", n_stories=1,
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_observability(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid observability"):
            await gen.generate_stories(
                sample_matrix, "mv_pharma_pro", "allies",
                observability="secret", n_stories=1,
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_power_dynamic(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        with pytest.raises(ValueError, match="Invalid power dynamic"):
            await gen.generate_stories(
                sample_matrix, "mv_pharma_pro", "allies",
                power_dynamic="chaotic", n_stories=1,
            )
