"""Uniform-random baseline player.

Returns a Action drawn uniformly from `game.legal_actions(state, self_idx)`.
Useful as a sanity-check opponent and for smoke-testing the harness end-
to-end without spending API/GPU on LLM calls.
"""

from __future__ import annotations

import random
from typing import Any

from ..base import Game


class RandomPlayer:
    def __init__(self, *, name: str = "Random", seed: int | None = None):
        self.name = name
        self._rng = random.Random(seed)

    def act(self, game: Game, state: Any, player_idx: int):
        legal = game.legal_actions(state, player_idx)
        if not legal:
            return {"type": "noop"}
        return self._rng.choice(legal)

    def receive_observation(self, obs: dict) -> None:
        # Random player has no memory.
        return
