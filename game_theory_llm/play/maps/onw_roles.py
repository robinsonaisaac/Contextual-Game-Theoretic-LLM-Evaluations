"""One Night Werewolf role data table (spec §5.1, §8 'onw_roles').

Plain data only — no game logic. Unit 1 (`one_night_werewolf.py`) consumes:
    ALL_ROLES, WAKE_ORDER, ROLE_COUNTS_BY_N, TEAM

Deck-size invariant: the deck dealt for an n-player match is always
``n_players + 3`` cards (n dealt to seats + 3 to the center). Therefore
``len(ROLE_COUNTS_BY_N[n]) == n + 3`` for every supported player count.

The role *strings* match the canonical names used in the existing game file
(e.g. "Werewolf", "Seer") so the two stay interoperable.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Full role roster (spec §5.1). Note: a role string appears here once even
# though a balanced deck may contain duplicates (e.g. two Werewolves, two
# Masons); duplication lives in ROLE_COUNTS_BY_N, not here.
# ---------------------------------------------------------------------------
ALL_ROLES: List[str] = [
    "Werewolf",
    "Minion",
    "Mason",
    "Seer",
    "Robber",
    "Troublemaker",
    "Insomniac",
    "Hunter",
    "Tanner",
    "Drunk",
    "Doppelganger",
    "Villager",
]

# ---------------------------------------------------------------------------
# Canonical night wake order. The role at the front wakes first.
# Doppelganger wakes first (it copies a role and may then act again in that
# role's slot); Insomniac wakes last (it must see its final, post-swap card).
# Roles with no night action (Mason aside, which only opens eyes to find the
# other Mason) such as Hunter, Tanner, Villager are absent from this list.
# WAKE_ORDER is a strict subset of ALL_ROLES (asserted in tests).
# ---------------------------------------------------------------------------
WAKE_ORDER: List[str] = [
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

# ---------------------------------------------------------------------------
# Team membership for win resolution.
#   "werewolf" -> wins with the werewolf team (no wolf must die)
#   "village"  -> wins if a werewolf dies
#   "tanner"   -> independent: wins iff the Tanner themself dies
# The Minion is on the werewolf team (wins with the wolves) even though it is
# not itself a Werewolf. The Doppelganger's effective team is determined at
# runtime by the role it copies; its *default* listing here is "village" so
# every role in ALL_ROLES has a concrete TEAM entry (the game overrides this
# once the copy target is known).
# ---------------------------------------------------------------------------
TEAM: Dict[str, str] = {
    "Werewolf": "werewolf",
    "Minion": "werewolf",
    "Mason": "village",
    "Seer": "village",
    "Robber": "village",
    "Troublemaker": "village",
    "Insomniac": "village",
    "Hunter": "village",
    "Tanner": "tanner",
    "Drunk": "village",
    "Doppelganger": "village",
    "Villager": "village",
}

# ---------------------------------------------------------------------------
# Balanced default role list (i.e. the deck) per player count, 3..10.
# len == n + 3 for each n. Built from standard ONW balance recommendations:
#   - always >= 2 Werewolves (so a lone-wolf bluff is possible and the wolf
#     is not always certain who its partner is),
#   - Seer / Robber / Troublemaker present in every deck (the core trio),
#   - Minion introduced at 6+, Masons at 7+, then Drunk / Insomniac / Hunter /
#     Tanner layered in as the deck grows,
#   - remaining slots padded with Villagers.
# Every deck contains at least one Werewolf (and from 6p also a Minion), so the
# werewolf team is always representable.
# ---------------------------------------------------------------------------
ROLE_COUNTS_BY_N: Dict[int, List[str]] = {
    3: [
        "Werewolf", "Werewolf",
        "Seer", "Robber", "Troublemaker",
        "Villager",
    ],  # 6
    4: [
        "Werewolf", "Werewolf",
        "Seer", "Robber", "Troublemaker",
        "Villager", "Villager",
    ],  # 7
    5: [
        "Werewolf", "Werewolf",
        "Seer", "Robber", "Troublemaker",
        "Villager", "Villager", "Villager",
    ],  # 8
    6: [
        "Werewolf", "Werewolf", "Minion",
        "Seer", "Robber", "Troublemaker",
        "Villager", "Villager", "Villager",
    ],  # 9
    7: [
        "Werewolf", "Werewolf", "Minion",
        "Mason", "Mason",
        "Seer", "Robber", "Troublemaker",
        "Villager", "Villager",
    ],  # 10
    8: [
        "Werewolf", "Werewolf", "Minion",
        "Mason", "Mason",
        "Seer", "Robber", "Troublemaker",
        "Drunk",
        "Villager", "Villager",
    ],  # 11
    9: [
        "Werewolf", "Werewolf", "Minion",
        "Mason", "Mason",
        "Seer", "Robber", "Troublemaker",
        "Insomniac", "Hunter",
        "Villager", "Villager",
    ],  # 12
    10: [
        "Werewolf", "Werewolf", "Minion",
        "Mason", "Mason",
        "Seer", "Robber", "Troublemaker",
        "Insomniac", "Hunter", "Tanner",
        "Villager", "Villager",
    ],  # 13
}
