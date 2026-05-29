"""Match-loop runner with JSONL logging.

The runner repeatedly:
  1. Asks the game who's next (`active_player`).
  2. Builds the prompt for that player (`render_prompt`).
  3. Calls `player.act(game, state, idx)`.
  4. Parses the player's response (`parse_action`) with retries.
  5. Steps the state and broadcasts an observation event to all players.
  6. Stops when `is_terminal(state)` is True.

Every event is appended to a JSONL log so matches can be replayed,
debugged, and aggregated offline.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List

from .base import Action, Game, MatchResult, ParseError, Player


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_dump(x):
    """JSON-safe coercion of game state for logging."""
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def run_match(
    game: Game,
    players: List[Player],
    *,
    seed: int,
    log_path: Path,
    max_turns: int = 1000,
    max_parse_retries: int = 2,
) -> MatchResult:
    """Run one match. Returns a `MatchResult`."""
    assert len(players) == game.n_players, (
        f"game expects {game.n_players} players, got {len(players)}"
    )
    rng = random.Random(seed)
    state = game.initial_state(rng)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(rec: dict) -> None:
        rec.setdefault("ts", _now())
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    log({
        "type": "match_start",
        "game": game.name,
        "n_players": game.n_players,
        "seed": seed,
        "players": [getattr(p, "name", type(p).__name__) for p in players],
    })

    turn = 0
    while not game.is_terminal(state) and turn < max_turns:
        active = game.active_player(state)
        if active < 0:
            # No one to act this micro-step; let the game advance phase.
            try:
                state = game.step(state, {"type": "advance_phase"})
                continue
            except Exception as e:
                log({"type": "advance_phase_error", "error": str(e)})
                break

        prompt = game.render_prompt(state, active)
        legal = game.legal_actions(state, active)
        log({
            "type": "prompt", "turn": turn, "player": active,
            "prompt": prompt, "n_legal": len(legal),
        })

        action: Action | None = None
        retries = 0
        last_err = ""
        while action is None and retries <= max_parse_retries:
            try:
                raw = players[active].act(game, state, active)
                if isinstance(raw, dict) and "type" in raw:
                    # Player returned a pre-parsed Action directly (rule
                    # bots and random bots take this shortcut).
                    action = raw
                else:
                    # Player returned text; parse it.
                    action = game.parse_action(state, active, str(raw))
            except ParseError as e:
                last_err = str(e)
                retries += 1
                log({"type": "parse_error", "turn": turn, "player": active,
                     "error": last_err, "retry": retries})
                # Re-prompt with the error embedded; chat players can pick
                # this up via receive_observation.
                players[active].receive_observation({
                    "type": "parse_error", "error": last_err,
                })

        if action is None:
            # Fall back to first legal action.
            action = legal[0] if legal else {"type": "noop"}
            log({"type": "fallback", "turn": turn, "player": active,
                 "action": action, "reason": last_err})

        log({
            "type": "action", "turn": turn, "player": active,
            "action": action,
        })

        try:
            state = game.step(state, action)
        except Exception as e:
            log({"type": "step_error", "turn": turn, "error": str(e),
                 "action": action})
            break

        # Broadcast the public action observation to all players.
        obs = {
            "type": "action", "turn": turn, "player": active,
            "action": action,
        }
        for p in players:
            try:
                p.receive_observation(obs)
            except Exception:
                pass

        turn += 1

    rewards = game.rewards(state) if game.is_terminal(state) else [0.0] * game.n_players
    log({"type": "terminal", "turn": turn, "rewards": rewards,
         "state": _safe_dump(state)})

    return MatchResult(
        rewards=list(rewards),
        log_path=str(log_path),
        terminal_state=state,
        n_turns=turn,
        metadata={"seed": seed, "game": game.name,
                  "players": [getattr(p, "name", type(p).__name__) for p in players]},
    )
