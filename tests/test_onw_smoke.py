"""Smoke-test the One Night Werewolf harness with 5 random players.

Verifies the runner can drive a full match to terminal with no exceptions,
the JSONL log contains the expected event types, and rewards sum to a
sensible value (one team wins).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from game_theory_llm.play import run_match
from game_theory_llm.play.games import OneNightWerewolf
from game_theory_llm.play.players import RandomPlayer


VALID_OUTCOMES = ("village", "werewolves", "nobody")


def test_random_players_finish_a_match():
    game = OneNightWerewolf()
    players = [RandomPlayer(seed=i, name=f"R{i}") for i in range(game.n_players)]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "match.jsonl"
        result = run_match(game, players, seed=42, log_path=log_path)
        assert result.terminal_state.phase == "terminal"
        assert result.terminal_state.winner_team in VALID_OUTCOMES
        # If a team won, the count of winners equals the number of seats on
        # that team; if no one won (rare edge case), all rewards are 0.
        if result.terminal_state.winner_team in ("village", "werewolves"):
            n_winners = sum(int(r > 0) for r in result.rewards)
            team_size = sum(
                1 for role in result.terminal_state.current_roles
                if (role == "Werewolf") == (result.terminal_state.winner_team == "werewolves")
            )
            assert n_winners == team_size, f"{n_winners=} {team_size=}"
        else:
            assert sum(result.rewards) == 0.0
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        kinds = [e.get("type") for e in events]
        assert "match_start" in kinds
        assert "terminal" in kinds
        assert any(k == "prompt" for k in kinds)
        assert any(k == "action" for k in kinds)


def test_multiple_seeds_dont_crash():
    """Run 16 matches with different seeds to flush out edge cases."""
    game = OneNightWerewolf()
    outcomes = []
    for seed in range(16):
        players = [RandomPlayer(seed=seed * 7 + i, name=f"R{i}")
                   for i in range(game.n_players)]
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / f"match_{seed}.jsonl"
            result = run_match(game, players, seed=seed, log_path=log_path)
            assert result.terminal_state.phase == "terminal"
            assert result.terminal_state.winner_team in VALID_OUTCOMES
            outcomes.append(result.terminal_state.winner_team)
    # At least one of {village, werewolves} should appear across 16 random
    # seeds so we know the resolution logic is hit.
    assert {"village", "werewolves"} & set(outcomes), \
        f"no proper resolution in 16 random matches; got {outcomes}"


if __name__ == "__main__":
    test_random_players_finish_a_match()
    test_multiple_seeds_dont_crash()
    print("ok")
