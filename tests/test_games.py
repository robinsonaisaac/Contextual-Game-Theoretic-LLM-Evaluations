# tests/test_games.py
"""Tests for game_theory_llm.games."""

import pytest

from game_theory_llm.games import (
    ALL_GAME_IDS,
    DEFAULT_GAME,
    GAME_REGISTRY,
    GameConfig,
    get_game,
)
from game_theory_llm.models import PayoffMatrix


class TestGameRegistry:
    def test_registry_has_exactly_7_games(self):
        assert len(GAME_REGISTRY) == 7

    def test_expected_game_ids(self):
        expected = {
            "prisoners_dilemma", "stag_hunt", "chicken", "pure_coordination",
            "harmony", "battle_of_the_sexes", "matching_pennies",
        }
        assert set(GAME_REGISTRY.keys()) == expected

    def test_all_game_ids_matches_registry(self):
        assert set(ALL_GAME_IDS) == set(GAME_REGISTRY.keys())

    def test_all_configs_are_game_config(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert isinstance(cfg, GameConfig), f"{gid} is not a GameConfig"


class TestGameConfig:
    def test_fields_non_empty(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert cfg.id, f"{gid} has empty id"
            assert cfg.name, f"{gid} has empty name"
            assert cfg.label_a, f"{gid} has empty label_a"
            assert cfg.label_b, f"{gid} has empty label_b"
            assert cfg.description, f"{gid} has empty description"

    def test_matrix_is_payoff_matrix(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert isinstance(cfg.matrix, PayoffMatrix), f"{gid} matrix is not PayoffMatrix"
            assert len(cfg.matrix.matrix) == 4

    def test_id_matches_registry_key(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert cfg.id == gid

    def test_focal_decision_is_a(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert cfg.focal_decision == "A", f"{gid} focal_decision is not 'A'"

    def test_frozen_immutable(self):
        cfg = GAME_REGISTRY["prisoners_dilemma"]
        with pytest.raises(AttributeError):
            cfg.name = "Modified"

    def test_nash_equilibria_are_tuples(self):
        for gid, cfg in GAME_REGISTRY.items():
            assert isinstance(cfg.nash_equilibria, tuple), f"{gid} nash is not tuple"


class TestPayoffOrdinals:
    """Verify payoff ordinal constraints for each canonical game."""

    def _payoffs(self, game_id):
        """Return (R, S, T, P) from the game matrix.

        Convention: AA=(R,R), AB=(S,T), BA=(T,S), BB=(P,P)
        where payoffs are for agent 1.
        """
        m = GAME_REGISTRY[game_id].matrix.matrix
        R = m[0][0]  # AA: both cooperate
        S = m[1][0]  # AB: agent1 cooperates, agent2 defects
        T = m[2][0]  # BA: agent1 defects, agent2 cooperates
        P = m[3][0]  # BB: both defect
        return R, S, T, P

    def test_prisoners_dilemma_T_gt_R_gt_P_gt_S(self):
        R, S, T, P = self._payoffs("prisoners_dilemma")
        assert T > R > P > S, f"PD ordinal violated: T={T}, R={R}, P={P}, S={S}"

    def test_stag_hunt_R_gt_T_gt_P_gt_S(self):
        R, S, T, P = self._payoffs("stag_hunt")
        assert R > T > P > S, f"SH ordinal violated: R={R}, T={T}, P={P}, S={S}"

    def test_chicken_T_gt_R_gt_S_gt_P(self):
        R, S, T, P = self._payoffs("chicken")
        assert T > R > S > P, f"Chicken ordinal violated: T={T}, R={R}, S={S}, P={P}"

    def test_pure_coordination_R_gt_P_gt_T_eq_S(self):
        R, S, T, P = self._payoffs("pure_coordination")
        assert R > P, f"Coord R>P violated: R={R}, P={P}"
        assert P > T, f"Coord P>T violated: P={P}, T={T}"
        assert T == S, f"Coord T==S violated: T={T}, S={S}"

    def test_harmony_R_gt_T_gt_S_gt_P(self):
        R, S, T, P = self._payoffs("harmony")
        assert R > T > S > P, f"Harmony ordinal violated: R={R}, T={T}, S={S}, P={P}"

    def test_harmony_cooperation_is_dominant(self):
        """In Harmony, A dominates B for both players."""
        m = GAME_REGISTRY["harmony"].matrix.matrix
        # Agent 1: A vs opponent A gives m[0][0]=4 > m[2][0]=3 (B vs opponent A)
        assert m[0][0] > m[2][0], "A should dominate B when opponent plays A"
        # Agent 1: A vs opponent B gives m[1][0]=2 > m[3][0]=1 (B vs opponent B)
        assert m[1][0] > m[3][0], "A should dominate B when opponent plays B"

    def test_battle_of_the_sexes_asymmetric_preferences(self):
        """Agent 1 prefers AA, agent 2 prefers BB."""
        m = GAME_REGISTRY["battle_of_the_sexes"].matrix.matrix
        assert m[0][0] > m[3][0], "Agent 1 should prefer AA over BB"
        assert m[3][1] > m[0][1], "Agent 2 should prefer BB over AA"
        # Miscoordination is worst for both
        assert m[0][0] > m[1][0], "AA > AB for agent 1"
        assert m[3][1] > m[1][1], "BB > AB for agent 2"

    def test_matching_pennies_zero_sum(self):
        """Constant-sum: each outcome sums to 1."""
        m = GAME_REGISTRY["matching_pennies"].matrix.matrix
        for i, (a1, a2) in enumerate(m):
            assert a1 + a2 == 1, f"Outcome {i} not constant-sum: {a1}+{a2}={a1+a2}"

    def test_matching_pennies_no_pure_nash(self):
        """Matching Pennies should have no pure-strategy Nash equilibria."""
        cfg = GAME_REGISTRY["matching_pennies"]
        assert cfg.nash_equilibria == ()


class TestGetGame:
    def test_returns_correct_config(self):
        cfg = get_game("stag_hunt")
        assert cfg.id == "stag_hunt"
        assert cfg.name == "Stag Hunt"

    def test_raises_for_unknown_id(self):
        with pytest.raises(ValueError, match="Unknown game"):
            get_game("rock_paper_scissors")


class TestDefaultGame:
    def test_default_is_prisoners_dilemma(self):
        assert DEFAULT_GAME.id == "prisoners_dilemma"

    def test_default_labels(self):
        assert DEFAULT_GAME.label_a == "Cooperate"
        assert DEFAULT_GAME.label_b == "Defect"
