"""Generate the schema-v2 viewer fixture log consumed by Unit 6 (the watcher).

This defines a tiny ``StubGame`` that composes ``MessagingMixin`` +
``AllianceMixin`` (messaging and alliances both enabled) over 4 ``RandomPlayer``
seats, runs a handful of negotiation + board phases, and writes
``tests/play/fixtures/sample_match.jsonl`` via the real ``run_match`` runner.

The goal is a *self-sufficient* log (spec §4 reconstruction guarantee): it must
contain, at minimum, every record type the viewer reads —
``match_start`` / ``setup`` / ``state_snapshot`` / ``observation`` (with
``message`` and ``alliance_event`` embedded obs) / ``phase_change`` /
``terminal``. The ``StubGame`` is intentionally *not* a real game: it exists
only to drive the shared messaging/alliance plumbing through the runner so the
fixture exercises the full schema.

Run directly to (re)generate the fixture::

    python3 tests/play/fixtures/make_fixture.py

It is deterministic: it sweeps candidate seeds and keeps the first one whose log
contains every required record type, so the checked-in fixture is stable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

# Allow running as a bare script (python3 tests/play/fixtures/make_fixture.py)
# as well as via pytest/imports.
if __package__ in (None, ""):
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from game_theory_llm.play import (  # noqa: E402
    AllianceMixin,
    AllianceState,
    GameConfig,
    MessagingMixin,
    NegotiationState,
    Obs,
    run_match,
)
from game_theory_llm.play.base import Game  # noqa: E402
from game_theory_llm.play.players import RandomPlayer  # noqa: E402


FIXTURE_PATH = Path(__file__).resolve().parent / "sample_match.jsonl"

# Phase names. The stub cycles: NEGOTIATION (talk + alliances) -> BOARD
# (a trivial "claim" action where honour/betray is judged) -> repeat for a
# couple of game-turns, then -> DONE (terminal).
PH_NEGO = "negotiation"
PH_BOARD = "board"
PH_DONE = "done"

# Number of full game-turns (nego + board) before terminating. Two turns keeps
# the fixture compact while still giving the uniform-random players enough
# opportunities to walk an alliance through its whole lifecycle (propose ->
# accept -> break) and to trigger honour/betray judgements at the board.
N_GAME_TURNS = 2


@dataclass
class _StubState:
    phase: str = PH_NEGO
    turn: int = 0                      # game-turn counter (alliance bookkeeping)
    game_turn: int = 0                 # how many board phases have resolved
    board_queue: List[int] = field(default_factory=list)  # seats to act on board
    claims: dict = field(default_factory=dict)            # seat -> "claim" str
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)
    done: bool = False
    winner: Any = None
    win_reason: str = ""


class StubGame(MessagingMixin, AllianceMixin, Game):
    """A throwaway 4-player game that wires the messaging + alliance mixins
    through the real runner so we can emit a representative schema-v2 log.

    Flow per game-turn:
      * ``negotiation`` sub-phase: every seat talks (say/whisper) and may
        propose / accept / decline / break alliances. Driven by the shared
        ``nego_*`` helpers; the negotiation auto-exits back to ``board`` after
        ``config.nego_rounds`` rounds.
      * ``board`` sub-phase: each seat makes a trivial public "claim". When a
        seat claims, ``judge_alliance`` decides whether that board action
        honours or betrays one of its active alliances (so the log carries
        ``honored`` / ``betrayed`` ``alliance_event`` records too).
    """

    name = "stub_game"
    n_players = 4

    # ------------------------------------------------------------- lifecycle
    def initial_state(self, rng) -> _StubState:
        st = _StubState()
        st.phase = PH_NEGO
        # Open the first negotiation; resume on the board phase afterwards.
        self.start_negotiation(st, return_phase=PH_BOARD)
        return st

    def _open_board(self, state: _StubState) -> None:
        state.phase = PH_BOARD
        state.board_queue = list(range(self.n_players))
        state.claims = {}

    # --------------------------------------------------------- runner-facing
    def active_player(self, state: _StubState) -> int:
        if state.done or state.phase == PH_DONE:
            return -1
        if state.phase == PH_NEGO:
            return self.nego_active_player(state)
        if state.phase == PH_BOARD:
            return state.board_queue[0] if state.board_queue else -1
        return -1

    def legal_actions(self, state: _StubState, player: int) -> List[dict]:
        if state.phase == PH_NEGO:
            return self.nego_legal_actions(state, player)
        if state.phase == PH_BOARD:
            # A single trivial board action: publicly claim a tile.
            return [{"type": "claim", "tile": player}]
        return []

    def render_prompt(self, state: _StubState, player: int) -> str:
        log = self.render_message_log(state, player)
        header = f"[stub] phase={state.phase} seat={player}"
        return header + ("\n" + log if log else "")

    def parse_action(self, state: _StubState, player: int, text: str) -> dict:
        if state.phase == PH_NEGO:
            return self.nego_parse(state, player, text)
        return {"type": "claim", "tile": player}

    def step(self, state: _StubState, action: dict) -> _StubState:
        t = action.get("type")

        # Phase advancement when nobody is left to act this micro-step.
        if t == "advance_phase":
            if state.phase == PH_BOARD and not state.board_queue:
                state.game_turn += 1
                state.turn += 1
                if state.game_turn >= N_GAME_TURNS:
                    state.phase = PH_DONE
                    state.done = True
                    state.winner = 0
                    state.win_reason = "fixture reached its turn cap"
                else:
                    # start_negotiation arms the NegotiationState but does not
                    # touch state.phase — the game owns the phase label, so set
                    # it here (and on each nego exit it is restored to PH_BOARD
                    # via the saved return_phase).
                    state.phase = PH_NEGO
                    self.start_negotiation(state, return_phase=PH_BOARD)
            return state

        if state.phase == PH_NEGO:
            # nego_step handles say/whisper/pass + alliance_* (the latter via
            # apply_alliance_action) and exits the negotiation when rounds run
            # out (setting state.phase = return_phase == PH_BOARD).
            self.nego_step(state, action)
            if state.phase == PH_BOARD and not state.board_queue:
                # Negotiation just closed; open the board for claims.
                self._open_board(state)
            return state

        if state.phase == PH_BOARD:
            seat = state.board_queue[0] if state.board_queue else -1
            if seat >= 0:
                state.claims[seat] = action.get("tile", seat)
                state.board_queue = state.board_queue[1:]
            return state

        return state

    def is_terminal(self, state: _StubState) -> bool:
        return state.done

    def rewards(self, state: _StubState) -> List[float]:
        # Winner (seat 0) gets 1.0, everyone else 0.0.
        return [1.0 if i == state.winner else 0.0 for i in range(self.n_players)]

    # ----------------------------------------------------------- observation
    def observations(self, prev_state, new_state, action, actor) -> List[Obs]:
        t = action.get("type")
        if t in ("say", "whisper", "pass_talk") or (t and t.startswith("alliance_")):
            return self.nego_observations(prev_state, new_state, action, actor)
        if t == "claim":
            # Public board move + any honour/betray judgement it triggers.
            obs_list: List[Obs] = [
                Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})
            ]
            for ev in self.judge_alliance(new_state, action, actor):
                cp = list(ev.get("counterparty", []))
                audience = sorted(set([actor]) | set(cp))
                obs_list.append(Obs(audience=audience, payload=ev))
            return obs_list
        return [Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

    # ----------------------------------------------------- alliance judgement
    def judge_alliance(self, state: _StubState, action, actor) -> List[dict]:
        """At a board ``claim``, the first active alliance ``actor`` belongs to
        is judged: even game-turns it is honoured, odd game-turns it is
        betrayed. This deterministically emits both event kinds across the run
        so the fixture exercises the honour/betray path."""
        from game_theory_llm.play import new_event

        alli: AllianceState = state.alli
        for al in alli.alliances.values():
            if al.status == "active" and actor in al.members:
                cp = [s for s in al.members if s != actor]
                event = "betrayed" if (state.game_turn % 2 == 1) else "honored"
                ev = new_event(event, al, turn=state.turn, actor=actor,
                               counterparty=cp, action_ref=dict(action))
                alli.events.append(ev)
                return [ev]
        return []

    # ----------------------------------------------------- watcher god-view
    def god_view(self, state: _StubState) -> dict:
        return {"note": "stub game has no hidden roles",
                "winner_is_always": 0}

    def snapshot(self, state: _StubState) -> dict:
        return {
            "public": {"phase": state.phase, "game_turn": state.game_turn,
                       "claims": dict(state.claims)},
            "hidden": {"n_alliances": len(state.alli.alliances)},
        }

    def render_board(self, state: _StubState, *, reveal: str = "god") -> str:
        claims = ", ".join(f"P{s}->{t}" for s, t in sorted(state.claims.items()))
        return f"[stub board] phase={state.phase} claims: {claims or '(none)'}"


# --------------------------------------------------------------------- driver
REQUIRED_TYPES = (
    "match_start", "setup", "state_snapshot", "observation",
    "phase_change", "terminal",
)

# Alliance lifecycle transitions the fixture must demonstrate (spec requires
# say/whisper/alliance_propose/accept/break to all appear in the log).
REQUIRED_ALLIANCE_EVENTS = ("propose", "accept", "break")
REQUIRED_ACTION_TYPES = ("say", "whisper")


def _alliance_event_kinds(events: List[dict]) -> set:
    out = set()
    for e in events:
        if e.get("type") == "observation":
            obs = e.get("obs") or {}
            if obs.get("type") == "alliance_event" and obs.get("event"):
                out.add(obs["event"])
    return out


def _action_kinds(events: List[dict]) -> set:
    return {e["action"].get("type") for e in events if e.get("type") == "action"}


def _embedded_obs_types(events: List[dict]) -> set:
    out = set()
    for e in events:
        if e.get("type") == "observation":
            ot = (e.get("obs") or {}).get("type")
            if ot:
                out.add(ot)
    return out


def _record_type_counts(events: List[dict]) -> dict:
    counts: dict = {}
    for e in events:
        counts[e.get("type")] = counts.get(e.get("type"), 0) + 1
    return counts


def _build_log(seed: int, out_path: Path) -> List[dict]:
    if out_path.exists():
        out_path.unlink()
    # whisper_visibility="metadata" so the log carries both message + message_meta.
    cfg = GameConfig(messaging=True, alliances=True, nego_rounds=2,
                     msgs_per_slot=2, whisper_visibility="metadata")
    game = StubGame(config=cfg)
    players = [RandomPlayer(name=f"P{i}", seed=seed * 100 + i)
               for i in range(game.n_players)]
    run_match(game, players, seed=seed, log_path=out_path, max_turns=400)
    return [json.loads(line) for line in out_path.read_text().splitlines()]


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = None
    # Sweep seeds; keep the first that yields every required record type, both
    # message + alliance_event embedded obs, AND the full propose/accept/break
    # alliance lifecycle plus say/whisper messages (RandomPlayer makes the exact
    # mix stochastic, so we pin a known-good seed for a stable checked-in log).
    for seed in range(1000):
        events = _build_log(seed, FIXTURE_PATH)
        types = {e.get("type") for e in events}
        embedded = _embedded_obs_types(events)
        alli_kinds = _alliance_event_kinds(events)
        act_kinds = _action_kinds(events)
        ok = (all(rt in types for rt in REQUIRED_TYPES)
              and "message" in embedded
              and "alliance_event" in embedded
              and all(ae in alli_kinds for ae in REQUIRED_ALLIANCE_EVENTS)
              and all(at in act_kinds for at in REQUIRED_ACTION_TYPES))
        if ok:
            best = (seed, events, embedded)
            break
    if best is None:
        raise SystemExit(
            "FAILED to produce a fixture containing all required record types "
            "+ the full alliance lifecycle in 1000 seeds — check StubGame / "
            "runner wiring.")

    seed, events, embedded = best
    counts = _record_type_counts(events)
    print(f"Wrote fixture: {FIXTURE_PATH}")
    print(f"  seed              : {seed}")
    print(f"  total records     : {len(events)}")
    print(f"  record-type counts: {json.dumps(counts, sort_keys=True)}")
    print(f"  embedded obs types: {sorted(embedded)}")
    print(f"  alliance events   : {sorted(_alliance_event_kinds(events))}")
    print(f"  action types      : {sorted(_action_kinds(events))}")
    missing = [rt for rt in REQUIRED_TYPES if rt not in counts]
    if missing:
        raise SystemExit(f"missing required record types: {missing}")
    print("  OK: all required record types present "
          "(incl. message + alliance_event observations).")


if __name__ == "__main__":
    main()
