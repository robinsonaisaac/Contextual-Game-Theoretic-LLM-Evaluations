# tests/conftest.py
"""Shared fixtures for the game_theory_llm test suite."""

from typing import Dict, Optional
from unittest.mock import AsyncMock

import pytest

from game_theory_llm.client import LLMClient, ModelConfig, DEFAULT_MODELS
from game_theory_llm.games import GAME_REGISTRY, get_game
from game_theory_llm.models import PayoffMatrix, Story


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockLLMClient:
    """A fake LLMClient that returns canned responses without network calls."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.models = dict(DEFAULT_MODELS)
        self._responses = responses or {
            "llama": "<analysis>Test analysis</analysis>\n<decision>A</decision>",
            "claude": "<analysis>Test analysis</analysis>\n<decision>B</decision>",
            "gpt4": "<analysis>Test analysis</analysis>\n<decision>A</decision>",
        }

    async def generate(self, prompt: str, model: str = "all") -> Dict[str, Optional[str]]:
        if model == "all":
            return dict(self._responses)
        return {model: self._responses.get(model)}

    async def generate_messages(self, messages: list, model: str = "all") -> Dict[str, Optional[str]]:
        if model == "all":
            return dict(self._responses)
        return {model: self._responses.get(model)}


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def mock_client_with_stories():
    """Client that returns story-wrapped content for generation tests."""
    content = (
        "<story>Story one about agents. <decision>A</decision></story>\n"
        "<story>Story two about agents. <decision>B</decision></story>"
    )
    return MockLLMClient(responses={
        "opus": content,
        "gpt-5.4-mini": content,
        "ds-v4-pro": content,
        "haiku": content,
        "gemini-flash": content,
        "llama": content,
        "deepseek": content,
        "claude": content,
        "gpt4": content,
    })


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_matrix():
    return PayoffMatrix([
        (3, 3),
        (0, 5),
        (5, 0),
        (1, 1),
    ])


@pytest.fixture
def sample_stories():
    return [
        Story(
            content="Agent Alpha and Agent Beta must decide. <decision>A</decision>",
            topic="mv_pharma_pro",
            actor_type="allies",
            observability="private",
            power_dynamic="symmetric",
            game_type="prisoners_dilemma",
            conversation_mode="single_turn",
        ),
        Story(
            content="Two nations face a choice. <decision>B</decision>",
            topic="mv_pharma_pro",
            actor_type="enemies",
            observability="public",
            power_dynamic="symmetric",
            game_type="prisoners_dilemma",
            conversation_mode="single_turn",
        ),
        Story(
            content="Friends at a crossroad. <decision>A</decision>",
            topic="pol_dem_rep",
            actor_type="allies",
            observability="private",
            power_dynamic="asymmetric",
            game_type="stag_hunt",
            conversation_mode="single_turn",
        ),
    ]


@pytest.fixture
def sample_game_config():
    return get_game("stag_hunt")
