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
    max_parse_retries: int = 3,
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

    # High-water mark of how many ``state.alli.events`` have been logged, so
    # each alliance judgement event is emitted exactly once (B7 dedupe).
    _alli_hw = {"n": 0}

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
            _drain_alliance_events(log, state, players, _alli_hw, turn)
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
        # `max_parse_retries` is a total-attempts budget (retry x3 == 3
        # tries total, not 3 retries after an initial try) — void the match
        # once that many consecutive parse failures have been logged.
        while action is None and retries < max_parse_retries:
            raw = ""
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
                     "error": last_err, "retry": retries,
                     "raw": str(raw)[:600]})
                # Re-prompt with the error embedded; chat players can pick this
                # up via receive_observation.
                players[active].receive_observation({
                    "type": "parse_error", "error": last_err,
                })

        if action is None:
            # Parse-void policy: no random substitution. Void the match and
            # record who failed — a condition that cannot produce parseable
            # output is itself a steering-dose signal.
            log({"type": "aborted", "reason": "unparseable_output",
                 "turn": turn, "player": active,
                 "model": getattr(players[active], "model_key",
                                  getattr(players[active], "name",
                                          type(players[active]).__name__)),
                 "steering": steering,
                 "phase": getattr(state, "phase", None),
                 "last_error": last_err, "last_raw": str(raw)[:600]})
            return MatchResult(
                rewards=[0.0] * game.n_players,
                log_path=str(log_path), terminal_state=state, n_turns=turn,
                metadata={"seed": seed, "game": game.name,
                          "match_id": match_id, "aborted": True,
                          "aborted_player": active,
                          "aborted_reason": "unparseable_output",
                          "players": [getattr(p, "name", type(p).__name__)
                                      for p in players]},
            )

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

        # Single-emitter for any alliance judgement events the game appended
        # during this step (B7 log records + M1 victim delivery).
        _drain_alliance_events(log, state, players, _alli_hw, turn)

        turn += 1

    # Drain any final alliance events appended on the terminal step so the
    # log and the terminal alliance_summary stay consistent.
    _drain_alliance_events(log, state, players, _alli_hw, turn)

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


# Events the games ALREADY deliver as observations (via their own
# ``observations``/``alliance_observations`` path). The runner logs every new
# alli event as a top-level ``alliance_event`` record, but only *delivers* the
# judgement events the games do NOT route, so victims learn of betrayal/honour
# without double-delivery (B7 / M1).
_GAME_DELIVERED_EVENTS = frozenset({"propose", "accept", "decline", "break"})


def _drain_alliance_events(log, state, players: List[Player], hw: dict,
                           turn: int) -> None:
    """Single-emitter for alliance accounting (B7 / M1).

    Scans ``state.alli.events`` for entries appended since the previous drain
    (tracked by the high-water mark ``hw['n']``). For each NEW event:
      * writes a top-level ``alliance_event`` log record (so the metrics
        replay path can recompute the §4 summary), and
      * for judgement events the games do not already route
        (``honored`` / ``betrayed`` / ``expired``), delivers an
        ``alliance_event`` observation to the affected seats (proposer +
        members + counterparty) so victims learn of betrayal — Risk in
        particular never delivered these.

    A state with no ``alli`` (StubGame / legacy) is a no-op, preserving
    backward compatibility.
    """
    alli = getattr(state, "alli", None)
    if alli is None:
        return
    events = getattr(alli, "events", None)
    if events is None:
        return
    start = hw.get("n", 0)
    n = len(events)
    if n <= start:
        return
    n_players = len(players)
    for ev in events[start:n]:
        # Top-level log record (additive: events already carry the canonical
        # alliance_event shape from new_event()).
        rec = dict(ev)
        rec["type"] = "alliance_event"
        rec.setdefault("turn", turn)
        log(rec)

        event = ev.get("event")
        if event in _GAME_DELIVERED_EVENTS:
            continue
        # Deliver judgement observations the games do not route themselves.
        members = ev.get("members", []) or []
        cp = ev.get("counterparty", []) or []
        proposer = ev.get("proposer")
        audience = set()
        for s in list(members) + list(cp):
            audience.add(int(s))
        if proposer is not None:
            audience.add(int(proposer))
        payload = {
            "type": "alliance_event",
            "event": event,
            "alliance_id": ev.get("alliance_id"),
            "kind": ev.get("kind"),
            "proposer": proposer,
            "members": list(members),
            "actor": ev.get("actor"),
            "counterparty": list(cp),
            "turn": ev.get("turn", turn),
        }
        for seat in sorted(audience):
            if seat < 0 or seat >= n_players:
                continue
            try:
                players[seat].receive_observation(dict(payload))
            except Exception:
                pass
    hw["n"] = n


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
