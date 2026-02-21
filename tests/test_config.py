# tests/test_config.py
"""Tests for game_theory_llm.config."""

import pytest

from game_theory_llm.config import (
    ACTOR_TYPES,
    ALL_TOPICS,
    ExperimentConfig,
    PRESETS,
    WORLD_DESCRIPTIONS,
    get_config,
)


class TestPresets:
    def test_full_preset_loads(self):
        cfg = get_config("full")
        assert len(cfg.topics) == 10
        assert "real_world" in cfg.world_types
        assert "imaginary_world" in cfg.world_types

    def test_small_preset_loads(self):
        cfg = get_config("small")
        assert len(cfg.topics) == 2
        assert cfg.world_types == ["real_world"]
        assert cfg.actor_types == ["allies"]

    def test_holdout_preset_loads(self):
        cfg = get_config("holdout")
        assert "Cybersecurity" in cfg.topics

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_config("nonexistent")


class TestExperimentConfig:
    def test_validate_valid(self):
        cfg = ExperimentConfig(
            topics=["business"],
            world_types=["real_world"],
            actor_types=["allies"],
        )
        cfg.validate()  # should not raise

    def test_validate_unknown_topic(self):
        cfg = ExperimentConfig(
            topics=["underwater basket weaving"],
            world_types=["real_world"],
            actor_types=["allies"],
        )
        with pytest.raises(ValueError, match="Unknown topic"):
            cfg.validate()

    def test_validate_unknown_world(self):
        cfg = ExperimentConfig(
            topics=["business"],
            world_types=["mars"],
            actor_types=["allies"],
        )
        with pytest.raises(ValueError, match="Unknown world type"):
            cfg.validate()

    def test_validate_unknown_actor(self):
        cfg = ExperimentConfig(
            topics=["business"],
            world_types=["real_world"],
            actor_types=["frenemies"],
        )
        with pytest.raises(ValueError, match="Unknown actor type"):
            cfg.validate()


class TestConstants:
    def test_all_topics_includes_holdout(self):
        assert "Cybersecurity" in ALL_TOPICS
        assert "international business" in ALL_TOPICS

    def test_world_descriptions_keys(self):
        assert "real_world" in WORLD_DESCRIPTIONS
        assert "imaginary_world" in WORLD_DESCRIPTIONS

    def test_actor_types_have_required_keys(self):
        for name, info in ACTOR_TYPES.items():
            assert "description" in info
            assert "types" in info
            assert isinstance(info["types"], list)
