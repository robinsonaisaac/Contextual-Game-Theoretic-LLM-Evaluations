"""Abstract Game / Player / Action interfaces for the multi-game play harness.

A `Game` is responsible for state, legal actions, prompt rendering, action
parsing, and rewards. The `state` is an opaque dataclass per game; the
runner does not look inside.

A `Player` is responsible for picking an `Action` given a game, a state,
and the player's seat index. The player CANNOT see global state directly
— they receive only the rendered prompt.

The `Runner` (see `runner.py`) owns the loop and the JSONL log.

Design notes:

- We pass the `Game` itself into `Player.act(game, state, player_idx)`
  so the player can call `game.render_prompt(state, player_idx)` to get
  the per-player view. This keeps hidden-info masking inside `Game`.
- Actions are dictionaries (kept simple to JSONify for logging) with a
  required `type` field plus game-specific payload.
- The runner handles parse retries: if `parse_action` raises
  `ParseError`, the player is re-prompted with the same context plus the
  error message, up to a small retry budget.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Protocol


class ParseError(Exception):
    """Raised by `Game.parse_action` when the player's text cannot be
    mapped to a legal action. The runner catches this and re-prompts."""


# An Action is a JSON-serialisable dict with at least a "type" key.
Action = dict


@dataclass
class MatchResult:
    """Final outcome of one match."""
    rewards: List[float]            # final per-player scalar reward
    log_path: str                   # absolute path to the JSONL log
    terminal_state: Any             # game-specific terminal state
    n_turns: int
    metadata: dict = field(default_factory=dict)


class Game(abc.ABC):
    """Abstract game. Implementations live in `game_theory_llm.play.games`."""

    name: str
    n_players: int

    @abc.abstractmethod
    def initial_state(self, rng) -> Any:
        """Return the initial game state given a `random.Random` instance."""

    @abc.abstractmethod
    def active_player(self, state) -> int:
        """Return the index of the player whose turn it is, or -1 if no one
        needs to act right now (the runner advances to the next phase)."""

    @abc.abstractmethod
    def legal_actions(self, state, player: int) -> List[Action]:
        """Enumerate legal actions for `player`. Returned for logging and
        for use by `parse_action`; the player itself sees only the prompt."""

    @abc.abstractmethod
    def render_prompt(self, state, player: int) -> str:
        """Build the per-player textual prompt. MUST mask hidden info that
        `player` is not entitled to see."""

    @abc.abstractmethod
    def parse_action(self, state, player: int, text: str) -> Action:
        """Parse the player's freeform text response into a legal Action.
        Raises `ParseError` on failure."""

    @abc.abstractmethod
    def step(self, state, action: Action) -> Any:
        """Apply `action` to `state` and return the new state."""

    @abc.abstractmethod
    def is_terminal(self, state) -> bool:
        """Return True iff the match has ended."""

    @abc.abstractmethod
    def rewards(self, state) -> List[float]:
        """Return per-player scalar rewards (only defined at terminal)."""


class Player(Protocol):
    """Anyone who can pick an Action given a Game and a player slot.

    `act` may be called many times per match (once per decision). The
    player is responsible for any internal memory (e.g., chat history).
    """
    name: str

    def act(self, game: Game, state: Any, player_idx: int) -> Action: ...

    def receive_observation(self, obs: dict) -> None:
        """Default no-op. The runner sends post-action observations
        (votes, role reveals, public events) to all players via this
        channel so chat-based players can append to their context."""
