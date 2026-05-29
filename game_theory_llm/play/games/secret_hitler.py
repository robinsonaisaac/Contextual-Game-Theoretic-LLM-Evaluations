"""Secret Hitler (5-player configuration).

A simplified but rules-correct implementation of the 5-player game:
- 3 Liberals, 1 Fascist, 1 Hitler.
- 5-6p rule: Hitler knows the Fascist (and vice versa).
- 17-card policy deck (6 Liberal + 11 Fascist), reshuffled when fewer than 3 cards remain.
- Standard election + policy enactment loop, with the failed-election tracker.
- Executive Actions:
    after 1st Fascist policy: none
    after 2nd Fascist policy: none
    after 3rd Fascist policy: Policy Peek (President sees top 3 cards; v0 keeps this internal)
    after 4th Fascist policy: Execution (President kills a player)
    after 5th Fascist policy: Execution + Veto Power (v0 ignores veto)

Win conditions:
  Liberals win iff
    - 5 Liberal policies enacted, OR
    - Hitler is executed.
  Fascists win iff
    - 6 Fascist policies enacted, OR
    - Hitler is elected Chancellor after >=3 Fascist policies have been enacted.

Action types emitted by parse_action:
  - {"type": "nominate", "target": int}
  - {"type": "vote",      "ja": bool}
  - {"type": "discard",   "index": int}        # 0..2, the policy to discard
  - {"type": "enact",     "index": int}        # 0..1, the policy to enact
  - {"type": "execute",   "target": int}       # at 4F/5F policy

The state stores both visible and hidden information; `render_prompt`
hides what each player isn't entitled to see (their role, plus the
Fascist-Hitler pairing, plus their own peeks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import Action, Game, ParseError


LIBERAL = "Liberal"
FASCIST = "Fascist"
HITLER = "Hitler"

# Phase tags
PH_NOMINATION = "nomination"
PH_VOTING = "voting"
PH_DISCARD = "discard"
PH_ENACT = "enact"
PH_EXECUTION = "execution"
PH_TERMINAL = "terminal"


@dataclass
class SHState:
    """Secret Hitler state."""
    n_players: int = 5
    roles: List[str] = field(default_factory=list)   # role per seat
    alive: List[bool] = field(default_factory=list)
    deck: List[str] = field(default_factory=list)     # remaining draw pile (top = end)
    discard: List[str] = field(default_factory=list)  # discarded policies
    enacted_liberal: int = 0
    enacted_fascist: int = 0
    election_tracker: int = 0
    president_idx: int = 0
    chancellor_idx: Optional[int] = None
    last_president: Optional[int] = None
    last_chancellor: Optional[int] = None
    phase: str = PH_NOMINATION

    # Per-round transient
    votes: Dict[int, bool] = field(default_factory=dict)
    drawn_policies: List[str] = field(default_factory=list)
    chancellor_policies: List[str] = field(default_factory=list)

    # Pending executive action
    pending_executive: Optional[str] = None  # None | "execution"
    execution_target_picked: Optional[int] = None

    # Outcome
    winner_team: Optional[str] = None
    win_reason: str = ""

    # Public log of major events
    public_events: List[str] = field(default_factory=list)


class SecretHitler(Game):
    name = "secret_hitler"
    n_players = 5

    LIBERAL_TO_WIN = 5
    FASCIST_TO_WIN = 6

    # ----------------------------------------------------------- setup
    def initial_state(self, rng) -> SHState:
        roles = [LIBERAL, LIBERAL, LIBERAL, FASCIST, HITLER]
        rng.shuffle(roles)
        deck = [LIBERAL] * 6 + [FASCIST] * 11
        rng.shuffle(deck)
        first_pres = rng.randint(0, self.n_players - 1)
        return SHState(
            n_players=self.n_players,
            roles=list(roles),
            alive=[True] * self.n_players,
            deck=list(deck),
            discard=[],
            enacted_liberal=0,
            enacted_fascist=0,
            election_tracker=0,
            president_idx=first_pres,
            chancellor_idx=None,
            phase=PH_NOMINATION,
        )

    # ------------------------------------------------------ active_player
    def active_player(self, state: SHState) -> int:
        if state.phase == PH_NOMINATION:
            return state.president_idx
        if state.phase == PH_VOTING:
            # Choose the next living player who hasn't voted yet.
            for i in range(self.n_players):
                if state.alive[i] and i not in state.votes:
                    return i
            return -1
        if state.phase == PH_DISCARD:
            return state.president_idx
        if state.phase == PH_ENACT:
            return state.chancellor_idx if state.chancellor_idx is not None else -1
        if state.phase == PH_EXECUTION:
            return state.president_idx
        return -1

    # ---------------------------------------------------- legal_actions
    def legal_actions(self, state: SHState, player: int) -> List[Action]:
        if state.phase == PH_NOMINATION:
            # Cannot nominate self, the last President, or the last Chancellor
            forbidden = {player}
            if state.last_president is not None: forbidden.add(state.last_president)
            if state.last_chancellor is not None: forbidden.add(state.last_chancellor)
            return [{"type": "nominate", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t not in forbidden]
        if state.phase == PH_VOTING:
            return [{"type": "vote", "ja": True}, {"type": "vote", "ja": False}]
        if state.phase == PH_DISCARD:
            return [{"type": "discard", "index": i} for i in range(3)]
        if state.phase == PH_ENACT:
            return [{"type": "enact", "index": i} for i in range(2)]
        if state.phase == PH_EXECUTION:
            return [{"type": "execute", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t != player]
        return []

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: SHState, player: int) -> str:
        head = self._public_header(state)
        my_info = self._private_header(state, player)
        if state.phase == PH_NOMINATION:
            body = self._nomination_prompt(state, player)
        elif state.phase == PH_VOTING:
            body = self._voting_prompt(state, player)
        elif state.phase == PH_DISCARD:
            body = self._discard_prompt(state, player)
        elif state.phase == PH_ENACT:
            body = self._enact_prompt(state, player)
        elif state.phase == PH_EXECUTION:
            body = self._execution_prompt(state, player)
        else:
            body = "[Game over.]"
        return f"{head}\n\n{my_info}\n\n{body}"

    def _public_header(self, state: SHState) -> str:
        alive = [i for i, a in enumerate(state.alive) if a]
        events = state.public_events[-12:]
        ev = "\n".join(f"  - {e}" for e in events) if events else "  (no events yet)"
        return (
            "=== Secret Hitler (5p) ===\n"
            f"Phase: {state.phase}\n"
            f"Living players: {alive}\n"
            f"Policies enacted — Liberal: {state.enacted_liberal}/5   Fascist: {state.enacted_fascist}/6\n"
            f"Election tracker: {state.election_tracker}/3\n"
            f"President: {state.president_idx}   Chancellor candidate: {state.chancellor_idx}\n"
            f"Recent events:\n{ev}\n"
            "Game rules:\n"
            "  - Liberals win by enacting 5 Liberal policies OR killing Hitler.\n"
            "  - Fascists win by enacting 6 Fascist policies OR electing Hitler\n"
            "    as Chancellor once at least 3 Fascist policies are enacted."
        )

    def _private_header(self, state: SHState, player: int) -> str:
        my_role = state.roles[player]
        bits = [f"You are player {player}. Your role is {my_role}."]
        # 5-6p rule: Hitler knows Fascist and Fascist knows Hitler.
        if my_role in (FASCIST, HITLER):
            for i, r in enumerate(state.roles):
                if i != player and r in (FASCIST, HITLER):
                    bits.append(f"You know: player {i} is on the Fascist team "
                                f"(specifically {r}).")
        return "\n".join(bits)

    @staticmethod
    def _nomination_prompt(state: SHState, player: int) -> str:
        forbidden = {state.last_president, state.last_chancellor}
        forbidden.discard(None)
        forbidden = sorted(x for x in forbidden if x is not None)
        return (
            f"You are President. Nominate a Chancellor from living players.\n"
            f"You cannot nominate yourself{', or recent gov members ' + str(forbidden) if forbidden else ''}.\n"
            "Respond with exactly:\n"
            "<nominate>X</nominate>   (X = player index)"
        )

    @staticmethod
    def _voting_prompt(state: SHState, player: int) -> str:
        return (
            f"Government on the ballot: President {state.president_idx}, "
            f"Chancellor {state.chancellor_idx}.\n"
            "Vote ja (yes) or nein (no).\n"
            "Respond with exactly one of:\n"
            "<vote>ja</vote>\n"
            "<vote>nein</vote>"
        )

    def _discard_prompt(self, state: SHState, player: int) -> str:
        # Show the President the 3 drawn policies.
        pols = ", ".join(f"({i}) {p}" for i, p in enumerate(state.drawn_policies))
        return (
            "President policy phase: you drew these 3 policies and must discard ONE.\n"
            f"Cards: {pols}\n"
            "Respond with exactly:\n"
            "<discard>I</discard>   (I in {0,1,2})"
        )

    def _enact_prompt(self, state: SHState, player: int) -> str:
        pols = ", ".join(f"({i}) {p}" for i, p in enumerate(state.chancellor_policies))
        return (
            "Chancellor policy phase: the President discarded one card; you must enact ONE of the remaining two.\n"
            f"Cards: {pols}\n"
            "Respond with exactly:\n"
            "<enact>I</enact>   (I in {0,1})"
        )

    @staticmethod
    def _execution_prompt(state: SHState, player: int) -> str:
        return (
            "Executive Action: as President, you must execute ONE living player.\n"
            "Respond with exactly:\n"
            "<execute>X</execute>   (X = player index, not you)"
        )

    # --------------------------------------------------------------- parse
    _NOM_RE = re.compile(r"<nominate>\s*(\d+)\s*</nominate>", re.I)
    _VOTE_RE = re.compile(r"<vote>\s*(ja|nein|yes|no|y|n)\s*</vote>", re.I)
    _DISCARD_RE = re.compile(r"<discard>\s*([012])\s*</discard>", re.I)
    _ENACT_RE = re.compile(r"<enact>\s*([01])\s*</enact>", re.I)
    _EXEC_RE = re.compile(r"<execute>\s*(\d+)\s*</execute>", re.I)

    def parse_action(self, state: SHState, player: int, text: str) -> Action:
        if state.phase == PH_NOMINATION:
            m = self._NOM_RE.search(text)
            if not m:
                raise ParseError("expected <nominate>X</nominate>")
            t = int(m.group(1))
            if not (0 <= t < self.n_players) or t == player or not state.alive[t]:
                raise ParseError(f"invalid nominee: {t}")
            forbidden = {state.last_president, state.last_chancellor}
            if t in forbidden:
                raise ParseError(f"player {t} was in the last government; not eligible")
            return {"type": "nominate", "target": t}
        if state.phase == PH_VOTING:
            m = self._VOTE_RE.search(text)
            if not m:
                raise ParseError("expected <vote>ja|nein</vote>")
            ja = m.group(1).lower() in ("ja", "yes", "y")
            return {"type": "vote", "ja": ja}
        if state.phase == PH_DISCARD:
            m = self._DISCARD_RE.search(text)
            if not m:
                raise ParseError("expected <discard>I</discard> with I in {0,1,2}")
            return {"type": "discard", "index": int(m.group(1))}
        if state.phase == PH_ENACT:
            m = self._ENACT_RE.search(text)
            if not m:
                raise ParseError("expected <enact>I</enact> with I in {0,1}")
            return {"type": "enact", "index": int(m.group(1))}
        if state.phase == PH_EXECUTION:
            m = self._EXEC_RE.search(text)
            if not m:
                raise ParseError("expected <execute>X</execute>")
            t = int(m.group(1))
            if t == player or not state.alive[t]:
                raise ParseError(f"invalid execution target: {t}")
            return {"type": "execute", "target": t}
        raise ParseError(f"no action expected in phase {state.phase}")

    # ----------------------------------------------------------------- step
    def step(self, state: SHState, action: Action) -> SHState:
        t = action.get("type")
        if state.phase == PH_NOMINATION and t == "nominate":
            state.chancellor_idx = int(action["target"])
            state.phase = PH_VOTING
            state.votes = {}
            return state

        if state.phase == PH_VOTING and t == "vote":
            # Vote belongs to the active player at this step
            voter = self.active_player(state)
            state.votes[voter] = bool(action["ja"])
            # Wait until all living have voted
            if all(state.alive[i] is False or i in state.votes
                   for i in range(self.n_players)):
                return self._resolve_vote(state)
            return state

        if state.phase == PH_DISCARD and t == "discard":
            idx = int(action["index"])
            removed = state.drawn_policies.pop(idx)
            state.discard.append(removed)
            state.chancellor_policies = list(state.drawn_policies)
            state.drawn_policies = []
            state.phase = PH_ENACT
            return state

        if state.phase == PH_ENACT and t == "enact":
            idx = int(action["index"])
            enacted = state.chancellor_policies.pop(idx)
            state.discard.append(state.chancellor_policies.pop(0))  # the remaining policy is also discarded
            state.public_events.append(
                f"Chancellor {state.chancellor_idx} enacted a {enacted} policy."
            )
            if enacted == LIBERAL:
                state.enacted_liberal += 1
            else:
                state.enacted_fascist += 1
            # Win checks
            if state.enacted_liberal >= self.LIBERAL_TO_WIN:
                state.winner_team = "liberal"; state.win_reason = "5 Liberal policies enacted"
                state.phase = PH_TERMINAL; return state
            if state.enacted_fascist >= self.FASCIST_TO_WIN:
                state.winner_team = "fascist"; state.win_reason = "6 Fascist policies enacted"
                state.phase = PH_TERMINAL; return state
            if (state.enacted_fascist >= 3
                    and state.chancellor_idx is not None
                    and state.roles[state.chancellor_idx] == HITLER):
                state.winner_team = "fascist"
                state.win_reason = f"Hitler (player {state.chancellor_idx}) elected Chancellor after 3 Fascist policies"
                state.phase = PH_TERMINAL; return state
            # Executive action triggers (5p schedule)
            if state.enacted_fascist in (4, 5):
                state.pending_executive = "execution"
                state.phase = PH_EXECUTION
                return state
            # Successful policy enacted; this government becomes term-limited
            return self._end_round(state, update_term_limits=True)

        if state.phase == PH_EXECUTION and t == "execute":
            target = int(action["target"])
            state.alive[target] = False
            killed_role = state.roles[target]
            state.public_events.append(
                f"President {state.president_idx} executed player {target}."
            )
            if killed_role == HITLER:
                state.winner_team = "liberal"
                state.win_reason = f"Hitler (player {target}) was killed"
                state.phase = PH_TERMINAL
                return state
            state.pending_executive = None
            return self._end_round(state, update_term_limits=False)

        return state

    def _resolve_vote(self, state: SHState) -> SHState:
        ja_count = sum(1 for v in state.votes.values() if v)
        nein_count = len(state.votes) - ja_count
        passed = ja_count > nein_count
        state.public_events.append(
            f"Vote on (P {state.president_idx}, C {state.chancellor_idx}): "
            f"ja={ja_count}, nein={nein_count} -> {'PASS' if passed else 'FAIL'}"
        )
        state.votes = {}
        if passed:
            # Hitler-elected-as-Chancellor-after-3F win check (immediate, before policy phase)
            if state.enacted_fascist >= 3 and state.roles[state.chancellor_idx] == HITLER:
                state.winner_team = "fascist"
                state.win_reason = (f"Hitler (player {state.chancellor_idx}) elected Chancellor "
                                    f"after 3 Fascist policies")
                state.phase = PH_TERMINAL
                return state
            # Draw 3 policies for President
            self._maybe_reshuffle(state)
            state.drawn_policies = [state.deck.pop() for _ in range(3)]
            state.phase = PH_DISCARD
            state.election_tracker = 0
            return state
        # Vote failed
        state.election_tracker += 1
        state.public_events.append(f"Election tracker at {state.election_tracker}/3")
        if state.election_tracker >= 3:
            # Chaos: top card enacts automatically, no executive action
            self._maybe_reshuffle(state)
            top = state.deck.pop()
            if top == LIBERAL:
                state.enacted_liberal += 1
            else:
                state.enacted_fascist += 1
            state.public_events.append(
                f"Chaos! Top card {top} enacted; election tracker reset; term limits forgotten."
            )
            state.election_tracker = 0
            state.last_president = None
            state.last_chancellor = None
            # Win checks
            if state.enacted_liberal >= self.LIBERAL_TO_WIN:
                state.winner_team = "liberal"; state.win_reason = "5 Liberal policies (via chaos)"
                state.phase = PH_TERMINAL; return state
            if state.enacted_fascist >= self.FASCIST_TO_WIN:
                state.winner_team = "fascist"; state.win_reason = "6 Fascist policies (via chaos)"
                state.phase = PH_TERMINAL; return state
        # Failed election: rotate president but do NOT impose term limits on
        # the rejected candidates.
        return self._end_round(state, update_term_limits=False)

    def _end_round(self, state: SHState, *, update_term_limits: bool) -> SHState:
        if update_term_limits:
            state.last_president = state.president_idx
            state.last_chancellor = state.chancellor_idx
        # Rotate President to next living player
        nxt = (state.president_idx + 1) % self.n_players
        while not state.alive[nxt]:
            nxt = (nxt + 1) % self.n_players
            if nxt == state.president_idx:
                break
        state.president_idx = nxt
        state.chancellor_idx = None
        state.phase = PH_NOMINATION
        return state

    @staticmethod
    def _maybe_reshuffle(state: SHState) -> None:
        if len(state.deck) < 3:
            import random as _r
            state.discard.extend(state.deck)
            state.deck = list(state.discard)
            state.discard = []
            # Use a deterministic shuffle from a per-state RNG-like seed.
            # We don't have access to the runner's rng here; use a hash of
            # the current state as a seed surrogate so behaviour is
            # reproducible per-run.
            seed = (state.enacted_fascist * 31 + state.enacted_liberal * 7
                    + state.election_tracker * 3 + len(state.public_events))
            _r.Random(seed).shuffle(state.deck)
            state.public_events.append("Policy deck reshuffled.")

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: SHState) -> bool:
        return state.phase == PH_TERMINAL

    def rewards(self, state: SHState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        out: List[float] = []
        for i, role in enumerate(state.roles):
            team = "liberal" if role == LIBERAL else "fascist"
            out.append(1.0 if team == state.winner_team else 0.0)
        return out
