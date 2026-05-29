"""End-to-end smoke test of the play harness: 4 games × random players.

Each game runs 8 matches with deterministic seeds and must:
  - reach a terminal phase
  - declare a winner (or 'nobody' in ONW's rare no-werewolves edge case)
  - produce a non-empty JSONL log with at least one prompt + action event

Designed to be cheap (~1 second per game) so it's safe to run in CI on
every commit touching the harness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from game_theory_llm.play import run_match
from game_theory_llm.play.games import (
    OneNightWerewolf, SecretHitler, RiskLite, DiplomacyLite,
)
from game_theory_llm.play.players import RandomPlayer


def _smoke(game, n_seeds: int = 8, n_players: int | None = None,
           max_turns: int = 800):
    np = n_players or game.n_players
    for seed in range(n_seeds):
        players = [RandomPlayer(seed=seed * 31 + i) for i in range(np)]
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / f"{game.name}__seed{seed}.jsonl"
            result = run_match(game, players, seed=seed, log_path=log_path,
                               max_turns=max_turns)
            # must reach terminal
            assert game.is_terminal(result.terminal_state), \
                f"{game.name} seed {seed}: did not terminate"
            # rewards are well-defined
            rewards = result.rewards
            assert len(rewards) == np
            for r in rewards:
                assert 0.0 <= r <= 1.0
            # at least one prompt + one action in the log
            events = [json.loads(l) for l in log_path.read_text().splitlines()]
            assert any(e.get("type") == "prompt" for e in events)
            assert any(e.get("type") == "action" for e in events)
            assert events[-1]["type"] == "terminal"


def test_onw():
    _smoke(OneNightWerewolf())


def test_secret_hitler():
    _smoke(SecretHitler(), max_turns=400)


def test_risk_lite():
    _smoke(RiskLite(), max_turns=400)


def test_diplomacy_lite():
    _smoke(DiplomacyLite(), max_turns=200)


if __name__ == "__main__":
    test_onw(); print("ONW OK")
    test_secret_hitler(); print("Secret Hitler OK")
    test_risk_lite(); print("Risk-Lite OK")
    test_diplomacy_lite(); print("Diplomacy-Lite OK")
    print("all 4 games smoke-passed")
