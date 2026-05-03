# tests/test_config.py
"""Tests for game_theory_llm.config."""

import pytest

from game_theory_llm.config import (
    ACTOR_TYPES,
    ALL_TOPIC_IDS,
    ALL_TOPICS,
    HOLDOUT_TOPIC_IDS,
    OBSERVABILITY,
    POWER_DYNAMIC,
    PRESETS,
    TOPICS,
    VALID_CONVERSATION_MODES,
    ExperimentConfig,
    Topic,
    get_config,
    get_matched_pairs,
    get_topics_by_axis,
)


class TestTopicStructure:
    def test_all_topics_are_topic_objects(self):
        for tid, topic in TOPICS.items():
            assert isinstance(topic, Topic), f"{tid} is not a Topic"

    def test_topic_ids_match_keys(self):
        for tid, topic in TOPICS.items():
            assert topic.id == tid

    def test_all_topic_ids_excludes_holdout(self):
        for tid in ALL_TOPIC_IDS:
            assert not TOPICS[tid].holdout

    def test_holdout_topic_ids_are_holdout(self):
        for tid in HOLDOUT_TOPIC_IDS:
            assert TOPICS[tid].holdout

    def test_all_topics_is_superset(self):
        assert set(ALL_TOPICS) == set(ALL_TOPIC_IDS) | set(HOLDOUT_TOPIC_IDS)

    def test_no_overlap_main_holdout(self):
        assert set(ALL_TOPIC_IDS) & set(HOLDOUT_TOPIC_IDS) == set()


class TestAxes:
    EXPECTED_AXES = {"moral_valence", "political", "temporal", "cultural", "baseline"}

    def test_all_expected_axes_present(self):
        axes = {t.axis for t in TOPICS.values()}
        assert self.EXPECTED_AXES <= axes

    def test_moral_valence_has_at_least_6_main(self):
        topics = get_topics_by_axis("moral_valence")
        assert len(topics) >= 6

    def test_political_has_at_least_6_main(self):
        topics = get_topics_by_axis("political")
        assert len(topics) >= 6

    def test_temporal_has_at_least_6_main(self):
        topics = get_topics_by_axis("temporal")
        assert len(topics) >= 6

    def test_cultural_has_at_least_6_main(self):
        topics = get_topics_by_axis("cultural")
        assert len(topics) >= 6

    def test_baseline_exists(self):
        topics = get_topics_by_axis("baseline")
        assert len(topics) >= 1

    def test_each_axis_has_holdout(self):
        for axis in ("moral_valence", "political", "temporal", "cultural"):
            holdouts = [
                t for t in get_topics_by_axis(axis, include_holdout=True)
                if t.holdout
            ]
            assert len(holdouts) >= 2, f"Axis {axis} has fewer than 2 holdout topics"


class TestMoralValenceMatchedPairs:
    def test_matched_pairs_exist(self):
        pairs = get_matched_pairs()
        assert len(pairs) >= 6

    def test_each_pair_has_prosocial_and_antisocial(self):
        for pro, anti in get_matched_pairs():
            assert pro.axis_value == "prosocial"
            assert anti.axis_value == "antisocial"

    def test_each_pair_shares_domain(self):
        for pro, anti in get_matched_pairs():
            assert pro.domain == anti.domain
            assert pro.domain != ""


class TestDimensions:
    def test_actor_types_binary(self):
        assert set(ACTOR_TYPES.keys()) == {"allies", "enemies"}

    def test_observability_binary(self):
        assert set(OBSERVABILITY.keys()) == {"private", "public"}

    def test_power_dynamic_binary(self):
        assert set(POWER_DYNAMIC.keys()) == {"symmetric", "asymmetric"}

    def test_actor_types_have_required_keys(self):
        for name, info in ACTOR_TYPES.items():
            assert "description" in info
            assert "types" in info
            assert isinstance(info["types"], list)

    def test_observability_values_are_strings(self):
        for key, val in OBSERVABILITY.items():
            assert isinstance(val, str)
            assert len(val) > 20

    def test_power_dynamic_values_are_strings(self):
        for key, val in POWER_DYNAMIC.items():
            assert isinstance(val, str)
            assert len(val) > 20


class TestPresets:
    def test_full_preset_loads(self):
        cfg = get_config("full")
        assert len(cfg.topics) == len(ALL_TOPIC_IDS)
        assert "allies" in cfg.actor_types
        assert "enemies" in cfg.actor_types
        assert "private" in cfg.observability
        assert "public" in cfg.observability

    def test_small_preset_loads(self):
        cfg = get_config("small")
        assert len(cfg.topics) >= 2
        assert "baseline_abstract" in cfg.topics

    def test_holdout_preset_loads(self):
        cfg = get_config("holdout")
        for tid in cfg.topics:
            assert TOPICS[tid].holdout

    def test_axis_presets_exist(self):
        for axis in ("moral_valence", "political", "temporal", "cultural"):
            cfg = get_config(axis)
            for tid in cfg.topics:
                assert TOPICS[tid].axis == axis

    def test_cross_game_preset_loads(self):
        cfg = get_config("cross_game")
        assert len(cfg.game_types) == 7
        assert "prisoners_dilemma" in cfg.game_types
        assert "stag_hunt" in cfg.game_types
        assert "chicken" in cfg.game_types
        assert "deadlock" in cfg.game_types
        assert "harmony" in cfg.game_types
        assert "battle_of_the_sexes" in cfg.game_types
        assert "matching_pennies" in cfg.game_types
        assert len(cfg.actor_types) == 1  # allies only

    def test_cross_game_full_preset_loads(self):
        cfg = get_config("cross_game_full")
        assert len(cfg.game_types) == 7
        assert len(cfg.actor_types) == 2

    def test_multi_turn_compare_preset_loads(self):
        cfg = get_config("multi_turn_compare")
        assert cfg.conversation_modes == ["single_turn", "multi_turn"]
        assert len(cfg.topics) == 6
        assert cfg.power_dynamic == ["symmetric"]

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_config("nonexistent")


class TestExperimentConfig:
    def test_validate_valid(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
        )
        cfg.validate()  # should not raise

    def test_validate_unknown_topic(self):
        cfg = ExperimentConfig(
            topics=["underwater basket weaving"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
        )
        with pytest.raises(ValueError, match="Unknown topic"):
            cfg.validate()

    def test_validate_unknown_actor(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["frenemies"],
            observability=["private"],
            power_dynamic=["symmetric"],
        )
        with pytest.raises(ValueError, match="Unknown actor type"):
            cfg.validate()

    def test_validate_unknown_observability(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["secret"],
            power_dynamic=["symmetric"],
        )
        with pytest.raises(ValueError, match="Unknown observability"):
            cfg.validate()

    def test_validate_unknown_power_dynamic(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["chaotic"],
        )
        with pytest.raises(ValueError, match="Unknown power dynamic"):
            cfg.validate()

    def test_validate_unknown_game_type(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
            game_types=["rock_paper_scissors"],
        )
        with pytest.raises(ValueError, match="Unknown game type"):
            cfg.validate()

    def test_conversation_modes_default(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
        )
        assert cfg.conversation_modes == ["single_turn"]

    def test_validate_unknown_conversation_mode(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
            conversation_modes=["three_turn"],
        )
        with pytest.raises(ValueError, match="Unknown conversation mode"):
            cfg.validate()

    def test_game_types_default(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
        )
        assert cfg.game_types == ["prisoners_dilemma"]

    def test_n_cells(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro", "mv_pharma_anti"],
            actor_types=["allies", "enemies"],
            observability=["private", "public"],
            power_dynamic=["symmetric", "asymmetric"],
        )
        assert cfg.n_cells == 2 * 2 * 2 * 2 * 1  # 1 game type (default)

    def test_n_cells_with_games(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro"],
            actor_types=["allies"],
            observability=["private"],
            power_dynamic=["symmetric"],
            game_types=["prisoners_dilemma", "stag_hunt"],
        )
        assert cfg.n_cells == 1 * 1 * 1 * 1 * 2 * 1

    def test_n_cells_with_conversation_modes(self):
        cfg = ExperimentConfig(
            topics=["mv_pharma_pro", "mv_pharma_anti"],
            actor_types=["allies", "enemies"],
            observability=["private"],
            power_dynamic=["symmetric"],
            conversation_modes=["single_turn", "multi_turn"],
        )
        assert cfg.n_cells == 2 * 2 * 1 * 1 * 1 * 2
