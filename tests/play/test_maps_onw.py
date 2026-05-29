"""Tests for One Night Werewolf role data (spec §5.1, §8 'onw_roles')."""

from __future__ import annotations

import pytest

from game_theory_llm.play.maps.onw_roles import (
    ALL_ROLES,
    ROLE_COUNTS_BY_N,
    TEAM,
    WAKE_ORDER,
)

VALID_TEAMS = {"werewolf", "village", "tanner"}
WOLF_TEAM_ROLES = {"Werewolf", "Minion"}


def test_every_role_has_a_team():
    for role in ALL_ROLES:
        assert role in TEAM, f"{role} missing a TEAM entry"
        assert TEAM[role] in VALID_TEAMS, f"{role} has invalid team {TEAM[role]!r}"


def test_team_has_no_extra_roles():
    # TEAM should map exactly the declared roster, nothing more.
    assert set(TEAM) == set(ALL_ROLES)


def test_all_roles_unique():
    assert len(ALL_ROLES) == len(set(ALL_ROLES))


def test_wake_order_is_subset_of_all_roles():
    assert set(WAKE_ORDER).issubset(set(ALL_ROLES))


def test_wake_order_has_no_duplicates():
    assert len(WAKE_ORDER) == len(set(WAKE_ORDER))


def test_wake_order_is_canonical():
    assert WAKE_ORDER == [
        "Doppelganger",
        "Werewolf",
        "Minion",
        "Mason",
        "Seer",
        "Robber",
        "Troublemaker",
        "Drunk",
        "Insomniac",
    ]


def test_role_counts_cover_3_to_10():
    assert set(ROLE_COUNTS_BY_N) == set(range(3, 11))


@pytest.mark.parametrize("n", list(range(3, 11)))
def test_deck_size_is_n_plus_3(n):
    deck = ROLE_COUNTS_BY_N[n]
    assert len(deck) == n + 3, f"deck for {n}p must be {n + 3} cards, got {len(deck)}"


@pytest.mark.parametrize("n", list(range(3, 11)))
def test_deck_entries_are_known_roles(n):
    for role in ROLE_COUNTS_BY_N[n]:
        assert role in ALL_ROLES, f"deck for {n}p contains unknown role {role!r}"


@pytest.mark.parametrize("n", list(range(3, 11)))
def test_deck_has_werewolf_or_minion(n):
    deck = ROLE_COUNTS_BY_N[n]
    assert WOLF_TEAM_ROLES.intersection(deck), (
        f"deck for {n}p must contain at least one Werewolf or Minion"
    )
