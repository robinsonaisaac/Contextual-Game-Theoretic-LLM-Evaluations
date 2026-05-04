# tests/test_generator.py
"""Tests for game_theory_llm.generator."""

import pytest

from game_theory_llm.games import get_game
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

    def test_game_config_injects_framing_hint(self, mock_client, sample_matrix):
        """All games use neutral 'Action A' / 'Action B' labels; the strategic
        flavor comes from the matrix and the framing_hint, NOT from semantic
        labels. The framing hint should be injected into the prompt."""
        gen = StoryGenerator(mock_client)
        game = get_game("stag_hunt")
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
            game_config=game,
        )
        # No "Hunt Stag" / "Hunt Hare" — labels are now neutral
        assert "Hunt Stag" not in prompt
        assert "Hunt Hare" not in prompt
        # But the framing hint should appear (the actual hint string, verbatim)
        assert game.framing_hint in prompt

    def test_game_config_overrides_matrix(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        game = get_game("stag_hunt")
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
            game_config=game,
        )
        # Stag Hunt payoffs (4,4) should appear, not PD (3,3)
        assert "4, 4" in prompt

    def test_no_game_config_uses_bare_labels(self, mock_client, sample_matrix):
        gen = StoryGenerator(mock_client)
        prompt = gen.create_query(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        # Should have bare "Decision A" without parenthetical
        assert "Decision A" in prompt
        assert "Decision A (" not in prompt


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_extracts_stories(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert len(result.stories) == 2

    @pytest.mark.asyncio
    async def test_decision_is_none_at_generation_time(self, mock_client_with_stories, sample_matrix):
        """Decisions are filled in by the analyzer when stories are sent to test
        subjects, not at generation time. The generated story body contains the
        elicitation template (with literal <decision>A</decision> / <decision>B</decision>
        tags as part of the question), so parsing decisions here would yield phantom
        values from the template.
        """
        gen = StoryGenerator(mock_client_with_stories)
        result = await gen.generate_batch(
            sample_matrix, "mv_pharma_pro", "allies",
        )
        assert result.stories[0].decision is None
        assert result.stories[1].decision is None

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


class TestMultiTurnGeneration:
    @pytest.mark.asyncio
    async def test_generate_story_multi_turn(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        story = await gen.generate_story_multi_turn(
            sample_matrix, "mv_pharma_pro", "allies", "llama",
        )
        assert story is not None
        assert story.conversation_mode == "multi_turn"
        assert story.conversation_history is not None
        assert len(story.conversation_history) == 4  # user/assistant/user/assistant

    @pytest.mark.asyncio
    async def test_generate_story_multi_turn_has_decision(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        story = await gen.generate_story_multi_turn(
            sample_matrix, "mv_pharma_pro", "allies", "llama",
        )
        assert story is not None
        assert story.decision in ("A", "B")

    @pytest.mark.asyncio
    async def test_generate_stories_multi_turn_mode(self, mock_client_with_stories, sample_matrix):
        gen = StoryGenerator(mock_client_with_stories)
        stories = await gen.generate_stories(
            sample_matrix, "mv_pharma_pro", "allies",
            n_stories=2, conversation_mode="multi_turn",
        )
        assert len(stories) > 0
        for s in stories:
            assert s.conversation_mode == "multi_turn"


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
