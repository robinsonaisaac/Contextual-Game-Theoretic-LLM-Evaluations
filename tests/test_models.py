# tests/test_models.py
"""Tests for game_theory_llm.models."""

import pytest
from datetime import datetime

from game_theory_llm.models import (
    AnalysisResult,
    BatchGenerationResult,
    PayoffMatrix,
    Story,
)


class TestPayoffMatrix:
    def test_valid_matrix(self):
        m = PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)])
        assert len(m.matrix) == 4

    def test_requires_exactly_4_scenarios(self):
        with pytest.raises(ValueError, match="exactly 4"):
            PayoffMatrix([(1, 1), (2, 2)])

    def test_format_matrix_contains_values(self):
        m = PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)])
        formatted = m.format_matrix()
        assert "3, 3" in formatted
        assert "0, 5" in formatted


class TestStory:
    def test_timestamp_auto_set(self):
        s = Story(content="test", topic="t", actor_type="a")
        assert isinstance(s.timestamp, datetime)

    def test_explicit_timestamp_preserved(self):
        ts = datetime(2024, 1, 1)
        s = Story(content="x", topic="t", actor_type="a", timestamp=ts)
        assert s.timestamp == ts

    def test_optional_fields_default_none(self):
        s = Story(content="x", topic="t", actor_type="a")
        assert s.prompt is None
        assert s.decision is None

    def test_default_dimensions(self):
        s = Story(content="x", topic="t", actor_type="a")
        assert s.observability == "private"
        assert s.power_dynamic == "symmetric"
        assert s.game_type == "prisoners_dilemma"

    def test_custom_dimensions(self):
        s = Story(
            content="x", topic="t", actor_type="a",
            observability="public", power_dynamic="asymmetric",
        )
        assert s.observability == "public"
        assert s.power_dynamic == "asymmetric"

    def test_custom_game_type(self):
        s = Story(
            content="x", topic="t", actor_type="a",
            game_type="stag_hunt",
        )
        assert s.game_type == "stag_hunt"

    def test_default_conversation_mode(self):
        s = Story(content="x", topic="t", actor_type="a")
        assert s.conversation_mode == "single_turn"

    def test_custom_conversation_mode(self):
        s = Story(
            content="x", topic="t", actor_type="a",
            conversation_mode="multi_turn",
        )
        assert s.conversation_mode == "multi_turn"

    def test_conversation_history_default_none(self):
        s = Story(content="x", topic="t", actor_type="a")
        assert s.conversation_history is None

    def test_conversation_history_stored(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        s = Story(
            content="x", topic="t", actor_type="a",
            conversation_history=history,
        )
        assert s.conversation_history == history
        assert len(s.conversation_history) == 2


class TestAnalysisResult:
    def test_construction(self, sample_stories):
        r = AnalysisResult(
            stories=sample_stories,
            decisions={"llama": ["A", "B", "A"]},
            summaries={"llama": ["summary"]},
            proportions={"llama": {"A": 0.67}},
            by_topic={},
            by_observability={},
            by_power={},
            by_actor={},
        )
        assert len(r.stories) == 3
        assert r.analysis_timestamp  # auto-set

    def test_timestamp_is_iso_format(self, sample_stories):
        r = AnalysisResult(
            stories=sample_stories,
            decisions={}, summaries={}, proportions={},
            by_topic={}, by_observability={}, by_power={}, by_actor={},
        )
        # Should be parseable as ISO datetime
        datetime.fromisoformat(r.analysis_timestamp)


class TestBatchGenerationResult:
    def test_construction(self):
        r = BatchGenerationResult(stories=[], summaries=[], unique_prompt="")
        assert r.stories == []
