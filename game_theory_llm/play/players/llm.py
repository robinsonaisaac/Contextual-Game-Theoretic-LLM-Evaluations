"""LLM player wrapping the project's OpenRouter-routed `LLMClient`.

Each `act()` call builds a fresh chat prompt from the per-player rendered
state (already masked by `Game.render_prompt`) plus a short conversation
history accumulated from `receive_observation` calls.

For multi-turn games (One Night Werewolf, Secret Hitler, Diplomacy) the
history is what gives the player memory between decisions; in single-
decision games it just contains the rendered prompt.

The returned value is the raw model text. `runner.run_match` will hand
it to `Game.parse_action`, which is responsible for mapping the prose
to a legal Action.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from ...client import LLMClient
from ..base import Game


class LLMPlayer:
    """Single-model LLM player."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        name: str | None = None,
        model_key: str = "claude-sonnet-4.6",
        system_prompt: str | None = None,
        max_history: int = 60,
    ):
        self.client = client or LLMClient()
        self.name = name or f"LLM[{model_key}]"
        self.model_key = model_key
        self.system_prompt = system_prompt or (
            "You are playing a strategic game. Read the situation carefully. "
            "Respond in the format the prompt requests."
        )
        self.history: List[dict] = []
        self.max_history = max_history

    def act(self, game: Game, state: Any, player_idx: int) -> str:
        prompt = game.render_prompt(state, player_idx)
        full_prompt = self._build_prompt(prompt)
        responses = asyncio.run(
            self.client.generate(full_prompt, model=self.model_key)
        )
        text = responses.get(self.model_key) or ""
        # Record both sides of the exchange in history.
        self._append("game", prompt)
        self._append(self.model_key, text)
        return text

    def receive_observation(self, obs: dict) -> None:
        """Append public events to the player's chat history."""
        text = self._format_obs(obs)
        if text:
            self._append("public", text)

    def _append(self, who: str, text: str) -> None:
        self.history.append({"who": who, "text": text})
        if len(self.history) > self.max_history:
            # Keep the most recent N to bound prompt growth.
            self.history = self.history[-self.max_history:]

    def _build_prompt(self, current_prompt: str) -> str:
        parts: List[str] = [self.system_prompt, ""]
        if self.history:
            parts.append("=== Game history (most recent last) ===")
            for h in self.history:
                tag = h["who"].upper()
                parts.append(f"[{tag}] {h['text']}")
            parts.append("=== End history ===")
            parts.append("")
        parts.append("=== Current situation ===")
        parts.append(current_prompt)
        return "\n".join(parts)

    @staticmethod
    def _format_obs(obs: dict) -> str:
        t = obs.get("type")
        if t == "action":
            who = obs.get("player")
            a = obs.get("action", {})
            return f"Public event: player {who} chose action {a}"
        if t == "parse_error":
            return f"Parse error on your last response: {obs.get('error')}. Re-read the response format requested and try again."
        if t == "phase_change":
            return f"Phase change: {obs.get('to')}"
        return ""
