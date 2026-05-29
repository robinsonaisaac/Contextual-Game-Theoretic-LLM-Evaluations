"""Match-loop runner with JSONL logging (schema v2).

The runner repeatedly:
  1. Asks the game who's next (`active_player`).
  2. Builds the prompt for that player (`render_prompt`).
  3. Calls `player.act(game, state, idx)`.
  4. Parses the player's response (`parse_action`) with retries.
  5. Steps the state, emits a phase-change diff (if the phase changed), and
     routes the game's `observations` to the entitled seats (masked) while
     logging the full god-view content.
  6. Stops when `is_terminal(state)` is True.

Every event is appended to a JSONL log so matches can be replayed, debugged,
and aggregated offline. Schema v2 stamps every record with a monotonic
`event_id` and a per-match `match_id`, and adds `setup`, `state_snapshot`,
`phase_change`, and `observation` records (spec §2, §4). v1 records keep their
shape; v2 only adds fields/types (INV-3).
"""

from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path
from typing import List

from .alliances import alliance_summary
from .base import GOD, Action, Game, MatchResult, Obs, ParseError, Player


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_dump(x):
    """JSON-safe coercion of game state for logging."""
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def _alliance_summary_or_empty(state) -> dict:
    """Reduce ``state.alli`` to the terminal alliance summary, or {}."""
    alli = getattr(state, "alli", None)
    if alli is None:
        return {}
    try:
        return alliance_summary(alli)
    except Exception:
        return {}


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

    match_id = uuid.uuid4().hex
    _eid = {"n": 0}

    def log(rec: dict) -> None:
        rec.setdefault("ts", _now())
        rec["event_id"] = _eid["n"]
        _eid["n"] += 1
        rec["match_id"] = match_id
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    cfg = getattr(game, "config", None)
    log({
        "type": "match_start", "schema": 2,
        "game": game.name, "n_players": game.n_players, "seed": seed,
        "players": [getattr(p, "name", type(p).__name__) for p in players],
        "config": (cfg.as_dict() if hasattr(cfg, "as_dict") else {}),
        "steering_tags": [getattr(p, "steering_tag", None) for p in players],
    })
    log({"type": "setup", "turn": 0, "god_view": game.god_view(state)})
    log({"type": "state_snapshot", "turn": 0,
         "phase": getattr(state, "phase", None), "snapshot": game.snapshot(state)})

    turn = 0
    while not game.is_terminal(state) and turn < max_turns:
        active = game.active_player(state)
        if active < 0:
            # No one to act this micro-step; let the game advance phase.
            prev_phase = getattr(state, "phase", None)
            try:
                state = game.step(state, {"type": "advance_phase"})
            except Exception as e:
                log({"type": "advance_phase_error", "error": str(e)})
                break
            _maybe_log_phase_change(log, game, state, players, prev_phase, turn)
            continue

        prompt = game.render_prompt(state, active)
        legal = game.legal_actions(state, active)
        steering = getattr(players[active], "steering_tag", None)
        log({
            "type": "prompt", "turn": turn, "player": active,
            "prompt": prompt, "n_legal": len(legal),
            "phase": getattr(state, "phase", None), "steering": steering,
        })

        action = None  # type: Action | None
        retries = 0
        last_err = ""
        while action is None and retries <= max_parse_retries:
            try:
                raw = players[active].act(game, state, active)
                if isinstance(raw, dict) and "type" in raw:
                    # Player returned a pre-parsed Action directly (rule bots
                    # and random bots take this shortcut).
                    action = raw
                else:
                    # Player returned text; parse it.
                    action = game.parse_action(state, active, str(raw))
            except ParseError as e:
                last_err = str(e)
                retries += 1
                log({"type": "parse_error", "turn": turn, "player": active,
                     "error": last_err, "retry": retries})
                # Re-prompt with the error embedded; chat players can pick this
                # up via receive_observation.
                players[active].receive_observation({
                    "type": "parse_error", "error": last_err,
                })

        if action is None:
            # Fall back to first legal action.
            action = legal[0] if legal else {"type": "noop"}
            log({"type": "fallback", "turn": turn, "player": active,
                 "action": action, "reason": last_err})

        action_rec = {
            "type": "action", "turn": turn, "player": active,
            "action": action, "phase": getattr(state, "phase", None),
            "steering": steering,
        }
        if isinstance(action, dict) and action.get("truncated"):
            action_rec["truncated"] = True
        log(action_rec)

        # Capture phase BEFORE step (in-place mutation safety — load-bearing).
        prev_phase = getattr(state, "phase", None)
        try:
            new_state = game.step(state, action)
        except Exception as e:
            log({"type": "step_error", "turn": turn, "error": str(e),
                 "action": action})
            break
        state = new_state

        _maybe_log_phase_change(log, game, state, players, prev_phase, turn)

        # Route the game's observations to entitled seats (masked), logging
        # the full god-view content (INV-2: the runner only routes).
        try:
            obs_list = game.observations(state, state, action, actor=active)
        except Exception:
            obs_list = [Obs(audience=list(range(game.n_players)),
                            payload={"type": "action", "player": active,
                                     "action": action})]
        for o in obs_list:
            logged = dict(o.log if o.log is not None else o.payload)
            logged.setdefault("turn", turn)
            log({"type": "observation", "turn": turn, "actor": active,
                 "audience": list(o.audience), "obs": logged})
            for seat in o.audience:
                if seat < 0:          # GOD/spectator pseudo-seat: never delivered
                    continue
                try:
                    payload = dict(o.payload)
                    payload.setdefault("turn", turn)
                    players[seat].receive_observation(payload)
                except Exception:
                    pass

        turn += 1

    rewards = game.rewards(state) if game.is_terminal(state) else [0.0] * game.n_players
    log({"type": "terminal", "turn": turn, "rewards": rewards,
         "state": _safe_dump(state),
         "winner": getattr(state, "winner",
                           getattr(state, "winner_team", None)),
         "win_reason": getattr(state, "win_reason", ""),
         "alliance_summary": _alliance_summary_or_empty(state)})

    return MatchResult(
        rewards=list(rewards),
        log_path=str(log_path),
        terminal_state=state,
        n_turns=turn,
        metadata={"seed": seed, "game": game.name, "match_id": match_id,
                  "players": [getattr(p, "name", type(p).__name__) for p in players]},
    )


def _maybe_log_phase_change(log, game: Game, state, players: List[Player],
                            prev_phase, turn: int) -> None:
    """Emit a ``phase_change`` diff + a refreshed ``state_snapshot`` and notify
    chat players whenever the phase changed after a step."""
    post_phase = getattr(state, "phase", None)
    if post_phase == prev_phase:
        return
    try:
        active_seat = game.active_player(state)
    except Exception:
        active_seat = -1
    log({"type": "phase_change", "turn": turn, "from": prev_phase,
         "to": post_phase, "active_seat": active_seat})
    log({"type": "state_snapshot", "turn": turn, "phase": post_phase,
         "snapshot": game.snapshot(state)})
    for p in players:
        try:
            p.receive_observation({"type": "phase_change", "to": post_phase})
        except Exception:
            pass
