"""Routing / masking correctness for the messaging layer + INV-1.

Two stub games:
  * ``LegacyStub`` overrides ONLY the 7 abstract methods (no ``observations``)
    and must broadcast every action to all players (INV-1 backward-compat).
  * ``ChatStub`` composes ``MessagingMixin`` and routes say/whisper through
    ``nego_observations`` so we can assert: a non-recipient never receives
    whisper text; the recipient + sender do; and the god-log carries the full
    text regardless.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from game_theory_llm.play import (
    GOD,
    GameConfig,
    MessagingMixin,
    NegotiationState,
    Obs,
    run_match,
)
from game_theory_llm.play.base import Game, ParseError


# --------------------------------------------------------------------- stubs
@dataclass
class _LegacyState:
    phase: str = "play"
    queue: List[int] = field(default_factory=lambda: [0, 1, 2])
    done: bool = False


class LegacyStub(Game):
    """Overrides ONLY the 7 abstract methods — no observations override.

    Each seat acts once; then terminal. The default Game.observations should
    broadcast each action to all players (legacy behaviour, INV-1).
    """

    name = "legacy_stub"
    n_players = 3

    def initial_state(self, rng) -> _LegacyState:
        return _LegacyState()

    def active_player(self, state) -> int:
        if state.done or not state.queue:
            return -1
        return state.queue[0]

    def legal_actions(self, state, player: int) -> List[dict]:
        return [{"type": "ping", "player": player}]

    def render_prompt(self, state, player: int) -> str:
        return f"seat {player}"

    def parse_action(self, state, player: int, text: str) -> dict:
        return {"type": "ping", "player": player}

    def step(self, state, action) -> _LegacyState:
        if action.get("type") == "advance_phase":
            state.done = True
            state.phase = "terminal"
            return state
        if state.queue:
            state.queue = state.queue[1:]
        if not state.queue:
            state.done = True
            state.phase = "terminal"
        return state

    def is_terminal(self, state) -> bool:
        return state.done

    def rewards(self, state) -> List[float]:
        return [0.0] * self.n_players


@dataclass
class _ChatState:
    phase: str = "nego"
    nego: NegotiationState = field(default_factory=NegotiationState)
    turn: int = 0


class ChatStub(MessagingMixin, Game):
    """Minimal messaging-enabled game used only to exercise routing/masking.

    We don't drive it through the full runner loop here; we call
    ``nego_observations`` directly with crafted actions.
    """

    name = "chat_stub"
    n_players = 4

    def initial_state(self, rng) -> _ChatState:
        st = _ChatState()
        self.start_negotiation(st, return_phase="play")
        return st

    # abstract-method satisfiers (unused by these unit assertions)
    def active_player(self, state) -> int:
        return self.nego_active_player(state)

    def legal_actions(self, state, player: int) -> List[dict]:
        return self.nego_legal_actions(state, player)

    def render_prompt(self, state, player: int) -> str:
        return self.render_message_log(state, player)

    def parse_action(self, state, player: int, text: str) -> dict:
        return self.nego_parse(state, player, text)

    def step(self, state, action) -> _ChatState:
        return self.nego_step(state, action)

    def is_terminal(self, state) -> bool:
        return state.phase == "terminal"

    def rewards(self, state) -> List[float]:
        return [0.0] * self.n_players

    def observations(self, prev_state, new_state, action, actor) -> List[Obs]:
        return self.nego_observations(prev_state, new_state, action, actor)


class _Recorder:
    """A player that records everything it receives."""

    def __init__(self, name: str):
        self.name = name
        self.received: List[dict] = []

    def act(self, game, state, player_idx):
        return {"type": "pass_talk"}

    def receive_observation(self, obs: dict) -> None:
        self.received.append(obs)


# --------------------------------------------------------------------- INV-1
def test_inv1_legacy_stub_broadcasts_to_all():
    game = LegacyStub()
    players = [_Recorder(f"P{i}") for i in range(game.n_players)]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "legacy.jsonl"
        result = run_match(game, players, seed=1, log_path=log_path)
        events = [json.loads(l) for l in log_path.read_text().splitlines()]

    # Every action observation must have been broadcast to all seats.
    obs_records = [e for e in events if e.get("type") == "observation"]
    assert obs_records, "no observation records emitted"
    for rec in obs_records:
        assert sorted(rec["audience"]) == list(range(game.n_players)), \
            f"legacy obs not broadcast to all: {rec['audience']}"
        assert rec["obs"]["type"] == "action"

    # Each recorder saw exactly the broadcast actions (3 seats act => 3 each).
    n_actions = len([e for e in events if e.get("type") == "action"])
    for p in players:
        action_obs = [o for o in p.received if o.get("type") == "action"]
        assert len(action_obs) == n_actions

    assert result.terminal_state.done


def test_inv1_default_observations_signature():
    """The default Game.observations broadcasts a public action obs."""
    game = LegacyStub()
    out = game.observations(None, None, {"type": "ping"}, actor=2)
    assert len(out) == 1
    o = out[0]
    assert sorted(o.audience) == [0, 1, 2]
    assert o.payload["type"] == "action"
    assert o.payload["player"] == 2
    assert GOD not in o.audience


# ----------------------------------------------------------------- masking
def test_public_say_reaches_all_living():
    game = ChatStub()
    state = game.initial_state(None)
    action = {"type": "say", "text": "hello everyone"}
    obs_list = game.nego_observations(state, state, action, actor=0)
    assert len(obs_list) == 1
    o = obs_list[0]
    assert sorted(o.audience) == [0, 1, 2, 3]
    assert o.payload["scope"] == "public"
    assert o.payload["text"] == "hello everyone"


def test_whisper_masks_bystanders_metadata():
    game = ChatStub()  # default whisper_visibility == "metadata"
    state = game.initial_state(None)
    # Seat 0 whispers to seat 1; seats 2,3 are bystanders.
    action = {"type": "whisper", "to": [1], "text": "secret plan"}
    obs_list = game.nego_observations(state, state, action, actor=0)

    # One full-text obs for {sender, recipient}, one meta obs for bystanders.
    full = [o for o in obs_list if o.payload.get("type") == "message"]
    meta = [o for o in obs_list if o.payload.get("type") == "message_meta"]
    assert len(full) == 1 and len(meta) == 1

    full_obs = full[0]
    assert sorted(full_obs.audience) == [0, 1]
    assert full_obs.payload["text"] == "secret plan"

    meta_obs = meta[0]
    assert sorted(meta_obs.audience) == [2, 3]
    # Bystanders must NOT receive the text.
    assert "text" not in meta_obs.payload
    assert meta_obs.payload["from"] == 0
    assert meta_obs.payload["n_recipients"] == 1
    # The god-log for the bystander obs still carries the full text.
    assert meta_obs.log is not None
    assert meta_obs.log["text"] == "secret plan"


def test_whisper_hidden_visibility_leaks_nothing():
    cfg = GameConfig(whisper_visibility="hidden")
    game = ChatStub(config=cfg)
    state = game.initial_state(None)
    action = {"type": "whisper", "to": [2], "text": "totally hidden"}
    obs_list = game.nego_observations(state, state, action, actor=0)
    # No message_meta obs at all; bystanders learn nothing.
    metas = [o for o in obs_list if o.payload.get("type") == "message_meta"]
    assert metas == []
    full = [o for o in obs_list if o.payload.get("type") == "message"]
    assert len(full) == 1
    assert sorted(full[0].audience) == [0, 2]


def test_render_message_log_masking():
    game = ChatStub()
    state = game.initial_state(None)
    # Seat 0 says public; seat 1 whispers seat 2.
    game.nego_step(state, {"type": "say", "text": "public hi"})
    # Recompute speaker bookkeeping: directly append a whisper to transcript via step.
    # Force the queue so seat 1 is the speaker for the whisper.
    state.nego.speak_queue = [1, 2, 3, 0]
    state.nego.budget = {0: 2, 1: 2, 2: 2, 3: 2}
    game.nego_step(state, {"type": "whisper", "to": [2], "text": "psst"})

    # Seat 0 (not in the whisper) sees only the public message.
    log0 = game.render_message_log(state, 0)
    assert "public hi" in log0
    assert "psst" not in log0

    # Seat 2 (recipient) sees both.
    log2 = game.render_message_log(state, 2)
    assert "public hi" in log2
    assert "psst" in log2

    # Seat 1 (sender of whisper) sees its own whisper.
    log1 = game.render_message_log(state, 1)
    assert "psst" in log1

    # Seat 3 (bystander) does NOT see the whisper text.
    log3 = game.render_message_log(state, 3)
    assert "psst" not in log3


def test_message_truncation_flag():
    cfg = GameConfig(max_msg_chars=10)
    game = ChatStub(config=cfg)
    state = game.initial_state(None)
    action = {"type": "say", "text": "x" * 50}
    game.nego_step(state, action)
    assert action.get("truncated") is True
    assert len(action["text"]) == 10


def test_whisper_to_self_rejected():
    game = ChatStub()
    state = game.initial_state(None)
    with pytest.raises(ParseError):
        game.nego_parse(state, 0, "<whisper to=0>hey</whisper>")


if __name__ == "__main__":
    test_inv1_legacy_stub_broadcasts_to_all()
    test_inv1_default_observations_signature()
    test_public_say_reaches_all_living()
    test_whisper_masks_bystanders_metadata()
    test_whisper_hidden_visibility_leaks_nothing()
    test_render_message_log_masking()
    test_message_truncation_flag()
    test_whisper_to_self_rejected()
    print("ok")
