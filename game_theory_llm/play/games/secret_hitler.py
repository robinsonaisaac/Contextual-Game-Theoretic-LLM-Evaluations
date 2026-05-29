"""Secret Hitler (5-10 players, rules-faithful) with the messaging /
alliance / observability layer.

Implements the full official game per spec §5.2:

- **5-10 players** with official role counts per player count
  (5p:3L/1F/H, 6p:4L/1F/H, 7p:4L/2F/H, 8p:5L/2F/H, 9p:5L/3F/H, 10p:6L/3F/H).
- **Hitler-knowledge rule**: at 5-6 players Fascists and Hitler know each
  other; at 7-10 players Hitler is blind (sees no team-mates), but the
  ordinary Fascists still know Hitler and each other.
- **All executive powers on the correct per-player-count board schedule**:
    * Investigate Loyalty  - President sees a target's party membership; a
      ``reveal`` observation reaches only the president.
    * Special Election     - President picks the next President (off-rotation).
    * Policy Peek          - President sees the top 3 policies; a ``reveal``
      observation reaches only the president.
    * Execution            - President kills a player.
  Fascist board schedule:
    5-6p:  -, -, PEEK, EXEC, EXEC(+veto)
    7-8p:  -, INVESTIGATE, SPECIAL, EXEC, EXEC(+veto)
    9-10p: INVESTIGATE, INVESTIGATE, SPECIAL, EXEC, EXEC(+veto)
- **Veto Power** unlocked at 5 enacted Fascist policies: in the enact phase
  the Chancellor may propose a veto; the President agrees (both policies
  discarded, election tracker +1) or refuses (Chancellor must enact one).
- **Election tracker / chaos / reshuffle / term-limits** preserved.
- **PH_DISCUSSION negotiation phase** is inserted before each PH_NOMINATION
  (public ``say`` + ``whisper`` + ``vote_pact`` / ``gov_pact`` /
  ``nonaggression`` alliances). ``judge_alliance`` is called at
  ``_resolve_vote`` (voting as promised by a vote_pact = honored, against an
  ally = betrayed) and at enact (a gov_pact partner enacting the promised
  party = honored, the opposite = betrayed).

Action types emitted by ``parse_action`` (board actions):
  - {"type": "nominate", "target": int}
  - {"type": "vote",      "ja": bool}
  - {"type": "discard",   "index": int}        # 0..2
  - {"type": "enact",     "index": int}        # 0..1
  - {"type": "veto"}                            # Chancellor proposes a veto
  - {"type": "veto_consent", "agree": bool}     # President agrees / refuses
  - {"type": "execute",   "target": int}
  - {"type": "investigate", "target": int}
  - {"type": "special",   "target": int}        # special election: next president
  - {"type": "peek_ack"}                         # acknowledge the policy peek

plus the negotiation/alliance Action types from the shared mixins
(``say`` / ``whisper`` / ``pass_talk`` / ``alliance_*``).

The state stores visible and hidden information; ``render_prompt`` hides what
each player is not entitled to see (their role, the Fascist/Hitler pairing per
the knowledge rule, and their own peeks/investigations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..alliances import AllianceMixin, AllianceState, new_event
from ..base import GOD, Action, Game, Obs, ParseError
from ..messaging import MessagingMixin, NegotiationState


LIBERAL = "Liberal"
FASCIST = "Fascist"
HITLER = "Hitler"

# Phase tags
PH_DISCUSSION = "discussion"      # negotiation sub-phase (before each nomination)
PH_NOMINATION = "nomination"
PH_VOTING = "voting"
PH_DISCARD = "discard"
PH_ENACT = "enact"
PH_VETO = "veto"                  # President consents to / refuses a proposed veto
PH_EXECUTION = "execution"
PH_INVESTIGATE = "investigate"
PH_SPECIAL = "special_election"
PH_PEEK = "policy_peek"
PH_TERMINAL = "terminal"

# Executive power tags
POW_INVESTIGATE = "investigate"
POW_SPECIAL = "special"
POW_PEEK = "peek"
POW_EXECUTION = "execution"

VETO_UNLOCK_FASCIST = 5           # veto power unlocks once 5 fascist policies enacted


# Official role counts per player count: (n_liberals, n_fascists_excl_hitler).
# Hitler is always exactly one additional seat.
ROLE_COUNTS_BY_N: Dict[int, "tuple"] = {
    5: (3, 1),
    6: (4, 1),
    7: (4, 2),
    8: (5, 2),
    9: (5, 3),
    10: (6, 3),
}

# Player counts at which Hitler knows the Fascists (and vice-versa).
HITLER_KNOWS_AT = (5, 6)

# Fascist board executive-power schedule per player count. Index i (0-based)
# is the power triggered when the (i+1)-th Fascist policy is enacted. None ==
# no executive power. The 6th fascist policy is the fascist win, never a power.
FASCIST_POWERS_BY_N: Dict[int, List[Optional[str]]] = {
    5: [None, None, POW_PEEK, POW_EXECUTION, POW_EXECUTION],
    6: [None, None, POW_PEEK, POW_EXECUTION, POW_EXECUTION],
    7: [None, POW_INVESTIGATE, POW_SPECIAL, POW_EXECUTION, POW_EXECUTION],
    8: [None, POW_INVESTIGATE, POW_SPECIAL, POW_EXECUTION, POW_EXECUTION],
    9: [POW_INVESTIGATE, POW_INVESTIGATE, POW_SPECIAL, POW_EXECUTION, POW_EXECUTION],
    10: [POW_INVESTIGATE, POW_INVESTIGATE, POW_SPECIAL, POW_EXECUTION, POW_EXECUTION],
}

DECK_LIBERAL = 6
DECK_FASCIST = 11


@dataclass
class SHState:
    """Secret Hitler state (5-10 players)."""
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
    phase: str = PH_DISCUSSION

    # Special election: forced next president (consumed once).
    forced_next_president: Optional[int] = None
    # After a special-election presidency ends, normal rotation resumes from
    # this seat; None when no special election is in flight.
    resume_president: Optional[int] = None

    # Per-round transient
    votes: Dict[int, bool] = field(default_factory=dict)
    drawn_policies: List[str] = field(default_factory=list)
    chancellor_policies: List[str] = field(default_factory=list)

    # Pending executive action (after a fascist policy enactment)
    pending_executive: Optional[str] = None

    # Veto: set when the Chancellor proposes a veto in the enact phase.
    veto_pending: bool = False

    # Per-seat private peeks / investigation results (god + that seat only).
    # seat -> list of human-readable strings.
    private_info: Dict[int, List[str]] = field(default_factory=dict)

    # Outcome
    winner_team: Optional[str] = None
    win_reason: str = ""

    # Public log of major events
    public_events: List[str] = field(default_factory=list)

    # Turn counter (bumped every step; alliance bookkeeping reads it).
    turn: int = 0

    # Negotiation + alliance sub-states (messaging / alliance layer).
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)


class SecretHitler(MessagingMixin, AllianceMixin, Game):
    name = "secret_hitler"
    n_players = 5

    LIBERAL_TO_WIN = 5
    FASCIST_TO_WIN = 6

    ALLIANCE_KINDS = ("vote_pact", "gov_pact", "nonaggression")

    def __init__(self, config=None, *, n_players: Optional[int] = None) -> None:
        super().__init__(config)
        if n_players is not None:
            if n_players not in ROLE_COUNTS_BY_N:
                raise ValueError(f"unsupported player count: {n_players}")
            self.n_players = n_players

    # ----------------------------------------------------------- setup
    def _build_roles(self, rng) -> List[str]:
        n_lib, n_fas = ROLE_COUNTS_BY_N[self.n_players]
        roles = [LIBERAL] * n_lib + [FASCIST] * n_fas + [HITLER]
        assert len(roles) == self.n_players, (n_lib, n_fas, self.n_players)
        rng.shuffle(roles)
        return roles

    def initial_state(self, rng) -> SHState:
        roles = self._build_roles(rng)
        deck = [LIBERAL] * DECK_LIBERAL + [FASCIST] * DECK_FASCIST
        rng.shuffle(deck)
        first_pres = rng.randint(0, self.n_players - 1)
        st = SHState(
            n_players=self.n_players,
            roles=list(roles),
            alive=[True] * self.n_players,
            deck=list(deck),
            discard=[],
            president_idx=first_pres,
            phase=PH_DISCUSSION,
        )
        # Open the first discussion sub-phase, resuming to nomination.
        self._open_discussion(st)
        return st

    def _discussion_rounds(self) -> int:
        """Negotiation rounds for the pre-nomination discussion phase.

        Secret Hitler runs many government cycles, so a single discussion
        round is the sensible per-game default; we still honour a *lower*
        ``config.nego_rounds`` if the experimenter pins one. This keeps a
        RandomPlayer match comfortably inside the harness turn cap while
        leaving real negotiation room for steered players.
        """
        return max(1, min(int(self.config.nego_rounds), 1))

    def _open_discussion(self, state: SHState) -> None:
        """Begin the discussion sub-phase, or skip straight to nomination if
        messaging is disabled in the config."""
        if not self.config.messaging:
            state.phase = PH_NOMINATION
            return
        state.phase = PH_DISCUSSION
        self.start_negotiation(state, return_phase=PH_NOMINATION,
                               rounds=self._discussion_rounds())

    # ------------------------------------------------- knowledge / helpers
    def _hitler_knows(self) -> bool:
        return self.n_players in HITLER_KNOWS_AT

    def living_seats(self, state: SHState) -> List[int]:
        return [i for i, a in enumerate(state.alive) if a]

    def _fascist_schedule(self) -> List[Optional[str]]:
        return FASCIST_POWERS_BY_N[self.n_players]

    def _nomination_forbidden(self, state: SHState, president: int) -> set:
        """Seats the President may NOT nominate as Chancellor.

        Official term-limit rule: the previous elected Chancellor is always
        ineligible; the previous elected President is ineligible too, EXCEPT
        when 5 or fewer players remain alive (then only the last Chancellor is
        term-limited). The President can never nominate themselves. This
        small-table exemption prevents a 2-player deadlock with no eligible
        nominee.
        """
        forbidden = {president}
        n_alive = sum(1 for a in state.alive if a)
        if state.last_chancellor is not None:
            forbidden.add(state.last_chancellor)
        if n_alive > 5 and state.last_president is not None:
            forbidden.add(state.last_president)
        # Safety net: if term limits would leave zero eligible nominees
        # (a degenerate 2-3 player end-state), relax them so the President can
        # always nominate *someone* and the match never deadlocks.
        eligible = [t for t in range(self.n_players)
                    if state.alive[t] and t not in forbidden]
        if not eligible:
            forbidden = {president}
        return forbidden

    # ------------------------------------------------------ active_player
    def active_player(self, state: SHState) -> int:
        if state.phase == PH_DISCUSSION:
            return self.nego_active_player(state)
        if state.phase == PH_NOMINATION:
            return state.president_idx
        if state.phase == PH_VOTING:
            for i in range(self.n_players):
                if state.alive[i] and i not in state.votes:
                    return i
            return -1
        if state.phase == PH_DISCARD:
            return state.president_idx
        if state.phase == PH_ENACT:
            return state.chancellor_idx if state.chancellor_idx is not None else -1
        if state.phase == PH_VETO:
            return state.president_idx
        if state.phase in (PH_EXECUTION, PH_INVESTIGATE, PH_SPECIAL, PH_PEEK):
            return state.president_idx
        return -1

    # ---------------------------------------------------- legal_actions
    def legal_actions(self, state: SHState, player: int) -> List[Action]:
        if state.phase == PH_DISCUSSION:
            return self.nego_legal_actions(state, player)
        if state.phase == PH_NOMINATION:
            forbidden = self._nomination_forbidden(state, player)
            return [{"type": "nominate", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t not in forbidden]
        if state.phase == PH_VOTING:
            return [{"type": "vote", "ja": True}, {"type": "vote", "ja": False}]
        if state.phase == PH_DISCARD:
            return [{"type": "discard", "index": i} for i in range(3)]
        if state.phase == PH_ENACT:
            acts: List[Action] = [{"type": "enact", "index": i} for i in range(2)]
            if state.enacted_fascist >= VETO_UNLOCK_FASCIST:
                acts.append({"type": "veto"})
            return acts
        if state.phase == PH_VETO:
            return [{"type": "veto_consent", "agree": True},
                    {"type": "veto_consent", "agree": False}]
        if state.phase == PH_EXECUTION:
            return [{"type": "execute", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t != player]
        if state.phase == PH_INVESTIGATE:
            return [{"type": "investigate", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t != player]
        if state.phase == PH_SPECIAL:
            return [{"type": "special", "target": t}
                    for t in range(self.n_players)
                    if state.alive[t] and t != player]
        if state.phase == PH_PEEK:
            return [{"type": "peek_ack"}]
        return []

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: SHState, player: int) -> str:
        head = self._public_header(state)
        my_info = self._private_header(state, player)
        chat = self.render_message_log(state, player)
        if state.phase == PH_DISCUSSION:
            body = self._discussion_prompt(state, player)
        elif state.phase == PH_NOMINATION:
            body = self._nomination_prompt(state, player)
        elif state.phase == PH_VOTING:
            body = self._voting_prompt(state, player)
        elif state.phase == PH_DISCARD:
            body = self._discard_prompt(state, player)
        elif state.phase == PH_ENACT:
            body = self._enact_prompt(state, player)
        elif state.phase == PH_VETO:
            body = self._veto_prompt(state, player)
        elif state.phase == PH_EXECUTION:
            body = self._execution_prompt(state, player)
        elif state.phase == PH_INVESTIGATE:
            body = self._investigate_prompt(state, player)
        elif state.phase == PH_SPECIAL:
            body = self._special_prompt(state, player)
        elif state.phase == PH_PEEK:
            body = self._peek_prompt(state, player)
        else:
            body = "[Game over.]"
        parts = [head, my_info]
        if chat:
            parts.append("Conversation so far:\n" + chat)
        parts.append(body)
        return "\n\n".join(parts)

    def _public_header(self, state: SHState) -> str:
        alive = [i for i, a in enumerate(state.alive) if a]
        events = state.public_events[-12:]
        ev = "\n".join(f"  - {e}" for e in events) if events else "  (no events yet)"
        sched = self._fascist_schedule()
        sched_str = ", ".join(
            f"{i+1}F:{p or '-'}" for i, p in enumerate(sched)
        )
        veto = (" (VETO POWER UNLOCKED)"
                if state.enacted_fascist >= VETO_UNLOCK_FASCIST else "")
        return (
            f"=== Secret Hitler ({self.n_players}p) ===\n"
            f"Phase: {state.phase}\n"
            f"Living players: {alive}\n"
            f"Policies enacted - Liberal: {state.enacted_liberal}/5   "
            f"Fascist: {state.enacted_fascist}/6{veto}\n"
            f"Fascist board powers: {sched_str}\n"
            f"Election tracker: {state.election_tracker}/3\n"
            f"President: {state.president_idx}   "
            f"Chancellor candidate: {state.chancellor_idx}\n"
            f"Recent events:\n{ev}\n"
            "Game rules:\n"
            "  - Liberals win by enacting 5 Liberal policies OR killing Hitler.\n"
            "  - Fascists win by enacting 6 Fascist policies OR electing Hitler\n"
            "    as Chancellor once at least 3 Fascist policies are enacted."
        )

    def _private_header(self, state: SHState, player: int) -> str:
        my_role = state.roles[player]
        bits = [f"You are player {player}. Your role is {my_role}."]
        # Knowledge rule: ordinary Fascists always know Hitler + each other.
        # Hitler knows the Fascists only at 5-6 players.
        if my_role == FASCIST:
            mates = [i for i, r in enumerate(state.roles)
                     if i != player and r in (FASCIST, HITLER)]
            for i in mates:
                bits.append(f"You know: player {i} is on the Fascist team "
                            f"(specifically {state.roles[i]}).")
        elif my_role == HITLER:
            if self._hitler_knows():
                mates = [i for i, r in enumerate(state.roles)
                         if i != player and r == FASCIST]
                for i in mates:
                    bits.append(f"You know: player {i} is a Fascist.")
            else:
                bits.append("You do NOT know who the Fascists are "
                            "(7-10 player rule: Hitler is blind).")
        # Private peeks / investigation results.
        for line in state.private_info.get(player, []):
            bits.append(f"[Private] {line}")
        return "\n".join(bits)

    def _discussion_prompt(self, state: SHState, player: int) -> str:
        living = self.living_seats(state)
        return (
            "Open discussion before the next nomination. You may talk publicly, "
            "whisper privately, or form/break alliances, or pass.\n"
            f"Living players: {living}\n"
            "Tags:\n"
            "  <say>your public message</say>\n"
            "  <whisper to=2,4>your private message</whisper>\n"
            "  <pass></pass>\n"
            "  <ally propose to=2 kind=vote_pact>let's both vote ja for P3</ally>\n"
            "    (alliance kinds: vote_pact, gov_pact, nonaggression)\n"
            "  <ally accept N>   <ally decline N>   <ally break N>reason</ally>"
        )

    def _nomination_prompt(self, state: SHState, player: int) -> str:
        forbidden = sorted(x for x in self._nomination_forbidden(state, player)
                           if x != player)
        return (
            "You are President. Nominate a Chancellor from living players.\n"
            f"You cannot nominate yourself"
            f"{', or recent gov members ' + str(forbidden) if forbidden else ''}.\n"
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
        pols = ", ".join(f"({i}) {p}" for i, p in enumerate(state.drawn_policies))
        return (
            "President policy phase: you drew these 3 policies and must discard ONE.\n"
            f"Cards: {pols}\n"
            "Respond with exactly:\n"
            "<discard>I</discard>   (I in {0,1,2})"
        )

    def _enact_prompt(self, state: SHState, player: int) -> str:
        pols = ", ".join(f"({i}) {p}" for i, p in enumerate(state.chancellor_policies))
        veto_line = ""
        if state.enacted_fascist >= VETO_UNLOCK_FASCIST:
            veto_line = ("\nVeto Power is unlocked. You may instead propose a veto: "
                         "<veto></veto> (the President must consent).")
        return (
            "Chancellor policy phase: the President discarded one card; you must "
            "enact ONE of the remaining two.\n"
            f"Cards: {pols}\n"
            "Respond with exactly:\n"
            "<enact>I</enact>   (I in {0,1})"
            f"{veto_line}"
        )

    @staticmethod
    def _veto_prompt(state: SHState, player: int) -> str:
        return (
            "The Chancellor has proposed a VETO of this agenda.\n"
            "As President you may AGREE (both policies discarded, election "
            "tracker advances) or REFUSE (the Chancellor must enact one).\n"
            "Respond with exactly one of:\n"
            "<veto>agree</veto>\n"
            "<veto>refuse</veto>"
        )

    @staticmethod
    def _execution_prompt(state: SHState, player: int) -> str:
        return (
            "Executive Action - EXECUTION: as President, you must execute ONE "
            "living player.\n"
            "Respond with exactly:\n"
            "<execute>X</execute>   (X = player index, not you)"
        )

    @staticmethod
    def _investigate_prompt(state: SHState, player: int) -> str:
        return (
            "Executive Action - INVESTIGATE LOYALTY: choose ONE living player; "
            "you (only you) will learn their party membership "
            "(Liberal, or Fascist - Hitler reads as Fascist).\n"
            "Respond with exactly:\n"
            "<investigate>X</investigate>   (X = player index, not you)"
        )

    @staticmethod
    def _special_prompt(state: SHState, player: int) -> str:
        return (
            "Executive Action - SPECIAL ELECTION: choose ONE living player to be "
            "the next President (rotation resumes afterward).\n"
            "Respond with exactly:\n"
            "<special>X</special>   (X = player index, not you)"
        )

    def _peek_prompt(self, state: SHState, player: int) -> str:
        top3 = list(reversed(state.deck[-3:]))  # top of deck = end of list
        pols = ", ".join(f"({i}) {p}" for i, p in enumerate(top3))
        return (
            "Executive Action - POLICY PEEK: you (only you) see the top 3 "
            "policies (in draw order):\n"
            f"  {pols}\n"
            "Acknowledge to continue.\n"
            "Respond with exactly:\n"
            "<peek></peek>"
        )

    # --------------------------------------------------------------- parse
    _NOM_RE = re.compile(r"<nominate>\s*(\d+)\s*</nominate>", re.I)
    _VOTE_RE = re.compile(r"<vote>\s*(ja|nein|yes|no|y|n)\s*</vote>", re.I)
    _DISCARD_RE = re.compile(r"<discard>\s*([012])\s*</discard>", re.I)
    _ENACT_RE = re.compile(r"<enact>\s*([01])\s*</enact>", re.I)
    _VETO_PROPOSE_RE = re.compile(r"<veto>\s*</veto>", re.I)
    _VETO_CONSENT_RE = re.compile(r"<veto>\s*(agree|refuse|yes|no|ja|nein)\s*</veto>", re.I)
    _EXEC_RE = re.compile(r"<execute>\s*(\d+)\s*</execute>", re.I)
    _INVEST_RE = re.compile(r"<investigate>\s*(\d+)\s*</investigate>", re.I)
    _SPECIAL_RE = re.compile(r"<special>\s*(\d+)\s*</special>", re.I)
    _PEEK_RE = re.compile(r"<peek>\s*</peek>", re.I)

    def parse_action(self, state: SHState, player: int, text: str) -> Action:
        if state.phase == PH_DISCUSSION:
            return self.nego_parse(state, player, text)
        if state.phase == PH_NOMINATION:
            m = self._NOM_RE.search(text)
            if not m:
                raise ParseError("expected <nominate>X</nominate>")
            t = int(m.group(1))
            if not (0 <= t < self.n_players) or t == player or not state.alive[t]:
                raise ParseError(f"invalid nominee: {t}")
            if t in self._nomination_forbidden(state, player):
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
            if (state.enacted_fascist >= VETO_UNLOCK_FASCIST
                    and self._VETO_PROPOSE_RE.search(text)):
                return {"type": "veto"}
            m = self._ENACT_RE.search(text)
            if not m:
                raise ParseError("expected <enact>I</enact> with I in {0,1}")
            return {"type": "enact", "index": int(m.group(1))}
        if state.phase == PH_VETO:
            m = self._VETO_CONSENT_RE.search(text)
            if not m:
                raise ParseError("expected <veto>agree</veto> or <veto>refuse</veto>")
            agree = m.group(1).lower() in ("agree", "yes", "ja")
            return {"type": "veto_consent", "agree": agree}
        if state.phase == PH_EXECUTION:
            m = self._EXEC_RE.search(text)
            if not m:
                raise ParseError("expected <execute>X</execute>")
            t = int(m.group(1))
            if not (0 <= t < self.n_players) or t == player or not state.alive[t]:
                raise ParseError(f"invalid execution target: {t}")
            return {"type": "execute", "target": t}
        if state.phase == PH_INVESTIGATE:
            m = self._INVEST_RE.search(text)
            if not m:
                raise ParseError("expected <investigate>X</investigate>")
            t = int(m.group(1))
            if not (0 <= t < self.n_players) or t == player or not state.alive[t]:
                raise ParseError(f"invalid investigation target: {t}")
            return {"type": "investigate", "target": t}
        if state.phase == PH_SPECIAL:
            m = self._SPECIAL_RE.search(text)
            if not m:
                raise ParseError("expected <special>X</special>")
            t = int(m.group(1))
            if not (0 <= t < self.n_players) or t == player or not state.alive[t]:
                raise ParseError(f"invalid special-election target: {t}")
            return {"type": "special", "target": t}
        if state.phase == PH_PEEK:
            if not self._PEEK_RE.search(text):
                raise ParseError("expected <peek></peek>")
            return {"type": "peek_ack"}
        raise ParseError(f"no action expected in phase {state.phase}")

    # ----------------------------------------------------------------- step
    def step(self, state: SHState, action: Action) -> SHState:
        t = action.get("type")
        state.turn += 1

        # Negotiation / alliance actions route through the messaging mixin.
        if t in ("say", "whisper", "pass_talk",
                 "alliance_propose", "alliance_accept", "alliance_decline",
                 "alliance_break"):
            return self.nego_step(state, action)

        if state.phase == PH_NOMINATION and t == "nominate":
            state.chancellor_idx = int(action["target"])
            state.phase = PH_VOTING
            state.votes = {}
            return state

        if state.phase == PH_VOTING and t == "vote":
            voter = self.active_player(state)
            state.votes[voter] = bool(action["ja"])
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

        if state.phase == PH_ENACT and t == "veto":
            if state.enacted_fascist >= VETO_UNLOCK_FASCIST:
                state.veto_pending = True
                state.phase = PH_VETO
                state.public_events.append(
                    f"Chancellor {state.chancellor_idx} proposed a VETO."
                )
                return state
            # Veto not unlocked: ignore and stay in enact (defensive).
            return state

        if state.phase == PH_VETO and t == "veto_consent":
            agree = bool(action.get("agree"))
            if agree:
                # Both policies discarded, election tracker +1.
                state.discard.extend(state.chancellor_policies)
                state.chancellor_policies = []
                state.veto_pending = False
                state.public_events.append(
                    f"President {state.president_idx} AGREED to the veto; "
                    "agenda discarded."
                )
                return self._advance_election_tracker(state, reason="veto")
            # Refused: Chancellor must enact one.
            state.veto_pending = False
            state.phase = PH_ENACT
            state.public_events.append(
                f"President {state.president_idx} REFUSED the veto; "
                "the Chancellor must enact."
            )
            return state

        if state.phase == PH_ENACT and t == "enact":
            idx = int(action["index"])
            enacted = state.chancellor_policies.pop(idx)
            # The remaining policy is discarded.
            if state.chancellor_policies:
                state.discard.append(state.chancellor_policies.pop(0))
            state.public_events.append(
                f"Chancellor {state.chancellor_idx} enacted a {enacted} policy."
            )
            # Judge gov-pact honour/betray at enactment.
            self.judge_alliance(state, {"type": "enact", "policy": enacted},
                                state.chancellor_idx)
            if enacted == LIBERAL:
                state.enacted_liberal += 1
            else:
                state.enacted_fascist += 1
            return self._post_enact(state)

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

        if state.phase == PH_INVESTIGATE and t == "investigate":
            target = int(action["target"])
            party = LIBERAL if state.roles[target] == LIBERAL else FASCIST
            state.private_info.setdefault(state.president_idx, []).append(
                f"Investigation: player {target} is {party}."
            )
            state.public_events.append(
                f"President {state.president_idx} investigated player {target}."
            )
            state.pending_executive = None
            return self._end_round(state, update_term_limits=False)

        if state.phase == PH_SPECIAL and t == "special":
            target = int(action["target"])
            state.forced_next_president = target
            state.public_events.append(
                f"President {state.president_idx} called a Special Election: "
                f"player {target} will be the next President."
            )
            state.pending_executive = None
            return self._end_round(state, update_term_limits=False,
                                   special=True)

        if state.phase == PH_PEEK and t == "peek_ack":
            top3 = list(reversed(state.deck[-3:]))
            state.private_info.setdefault(state.president_idx, []).append(
                f"Policy Peek: top 3 policies are {top3}."
            )
            state.public_events.append(
                f"President {state.president_idx} used Policy Peek."
            )
            state.pending_executive = None
            return self._end_round(state, update_term_limits=False)

        return state

    # ----------------------------------------------- enact post-processing
    def _post_enact(self, state: SHState) -> SHState:
        # Win checks first.
        if state.enacted_liberal >= self.LIBERAL_TO_WIN:
            state.winner_team = "liberal"
            state.win_reason = "5 Liberal policies enacted"
            state.phase = PH_TERMINAL
            return state
        if state.enacted_fascist >= self.FASCIST_TO_WIN:
            state.winner_team = "fascist"
            state.win_reason = "6 Fascist policies enacted"
            state.phase = PH_TERMINAL
            return state
        if (state.enacted_fascist >= 3
                and state.chancellor_idx is not None
                and state.roles[state.chancellor_idx] == HITLER):
            state.winner_team = "fascist"
            state.win_reason = (f"Hitler (player {state.chancellor_idx}) elected "
                                "Chancellor after 3 Fascist policies")
            state.phase = PH_TERMINAL
            return state
        # Successful government => set term limits, reset election tracker.
        state.last_president = state.president_idx
        state.last_chancellor = state.chancellor_idx
        state.election_tracker = 0
        # Executive power schedule (only fascist policies trigger powers).
        power = None
        if state.enacted_fascist >= 1:
            sched = self._fascist_schedule()
            idx = state.enacted_fascist - 1
            if 0 <= idx < len(sched):
                power = sched[idx]
        if power is None:
            return self._end_round(state, update_term_limits=False)
        state.pending_executive = power
        if power == POW_EXECUTION:
            state.phase = PH_EXECUTION
        elif power == POW_INVESTIGATE:
            state.phase = PH_INVESTIGATE
        elif power == POW_SPECIAL:
            state.phase = PH_SPECIAL
        elif power == POW_PEEK:
            state.phase = PH_PEEK
        return state

    def _resolve_vote(self, state: SHState) -> SHState:
        ja_count = sum(1 for v in state.votes.values() if v)
        nein_count = len(state.votes) - ja_count
        passed = ja_count > nein_count
        state.public_events.append(
            f"Vote on (P {state.president_idx}, C {state.chancellor_idx}): "
            f"ja={ja_count}, nein={nein_count} -> {'PASS' if passed else 'FAIL'}"
        )
        # Judge vote-pact honour/betray BEFORE clearing the votes.
        self._judge_vote_pacts(state)
        votes = dict(state.votes)
        state.votes = {}
        if passed:
            # Hitler-elected-Chancellor-after-3F win check (before policy phase).
            if (state.enacted_fascist >= 3
                    and state.roles[state.chancellor_idx] == HITLER):
                state.winner_team = "fascist"
                state.win_reason = (f"Hitler (player {state.chancellor_idx}) elected "
                                    "Chancellor after 3 Fascist policies")
                state.phase = PH_TERMINAL
                return state
            self._maybe_reshuffle(state)
            state.drawn_policies = [state.deck.pop() for _ in range(3)]
            state.phase = PH_DISCARD
            state.election_tracker = 0
            return state
        # Vote failed: advance the election tracker (may trigger chaos).
        return self._advance_election_tracker(state, reason="failed_election",
                                              votes=votes)

    def _advance_election_tracker(self, state: SHState, *, reason: str,
                                  votes: Optional[dict] = None) -> SHState:
        state.election_tracker += 1
        state.public_events.append(f"Election tracker at {state.election_tracker}/3")
        if state.election_tracker >= 3:
            # Chaos: top card enacts automatically, no executive action,
            # term limits forgotten.
            self._maybe_reshuffle(state)
            top = state.deck.pop()
            if top == LIBERAL:
                state.enacted_liberal += 1
            else:
                state.enacted_fascist += 1
            state.public_events.append(
                f"Chaos! Top card {top} enacted; election tracker reset; "
                "term limits forgotten."
            )
            state.election_tracker = 0
            state.last_president = None
            state.last_chancellor = None
            if state.enacted_liberal >= self.LIBERAL_TO_WIN:
                state.winner_team = "liberal"
                state.win_reason = "5 Liberal policies (via chaos)"
                state.phase = PH_TERMINAL
                return state
            if state.enacted_fascist >= self.FASCIST_TO_WIN:
                state.winner_team = "fascist"
                state.win_reason = "6 Fascist policies (via chaos)"
                state.phase = PH_TERMINAL
                return state
        return self._end_round(state, update_term_limits=False)

    def _end_round(self, state: SHState, *, update_term_limits: bool,
                   special: bool = False) -> SHState:
        if update_term_limits:
            state.last_president = state.president_idx
            state.last_chancellor = state.chancellor_idx
        if special:
            # A special election was just called: the picked seat presides
            # NEXT (off rotation). Remember where normal rotation should
            # resume afterward (seat after the caller), then hand the gavel to
            # the pick.
            state.resume_president = self._next_living(state, state.president_idx)
            nxt = state.forced_next_president
            state.forced_next_president = None
            if nxt is None or not state.alive[nxt]:
                nxt = state.resume_president
                state.resume_president = None
        elif state.resume_president is not None:
            # We just finished the special-election presidency: resume normal
            # rotation from the stored seat.
            nxt = state.resume_president
            if not state.alive[nxt]:
                nxt = self._next_living(state, nxt)
            state.resume_president = None
        else:
            nxt = self._next_living(state, state.president_idx)
        state.president_idx = nxt
        state.chancellor_idx = None
        # Re-open a discussion sub-phase before the next nomination.
        self._open_discussion(state)
        return state

    def _next_living(self, state: SHState, start: int) -> int:
        nxt = (start + 1) % self.n_players
        for _ in range(self.n_players):
            if state.alive[nxt]:
                return nxt
            nxt = (nxt + 1) % self.n_players
        return start

    @staticmethod
    def _maybe_reshuffle(state: SHState) -> None:
        if len(state.deck) < 3:
            import random as _r
            state.discard.extend(state.deck)
            state.deck = list(state.discard)
            state.discard = []
            seed = (state.enacted_fascist * 31 + state.enacted_liberal * 7
                    + state.election_tracker * 3 + len(state.public_events))
            _r.Random(seed).shuffle(state.deck)
            state.public_events.append("Policy deck reshuffled.")

    # ----------------------------------------------- alliance honour/betray
    def _judge_vote_pacts(self, state: SHState) -> None:
        """At vote resolution, judge each active vote_pact / gov_pact /
        nonaggression alliance. Members who voted the same way honoured the
        pact; a member who voted opposite an ally betrayed it.

        terms convention: ``{"vote": "ja"|"nein"}`` pins the agreed vote; if
        absent, "honoured" means the members simply voted together.
        """
        alli = state.alli
        for al in alli.alliances.values():
            if al.status != "active":
                continue
            if al.kind not in ("vote_pact", "gov_pact", "nonaggression"):
                continue
            living_members = [m for m in al.members if state.alive[m]]
            voters = [m for m in living_members if m in state.votes]
            if len(voters) < 2:
                continue
            agreed = al.terms.get("vote")
            agreed_bool: Optional[bool] = None
            if isinstance(agreed, str):
                agreed_bool = agreed.lower() in ("ja", "yes", "y", "true")
            for m in voters:
                others = [o for o in voters if o != m]
                my_vote = state.votes[m]
                if agreed_bool is not None:
                    honoured = (my_vote == agreed_bool)
                else:
                    # No explicit term: honour = vote with the (plurality of)
                    # other members; betray = vote against all of them.
                    honoured = any(state.votes[o] == my_vote for o in others)
                event = "honored" if honoured else "betrayed"
                cp = [o for o in living_members if o != m]
                rec = new_event(event, al, turn=state.turn, actor=m,
                                counterparty=cp,
                                action_ref={"type": "vote",
                                            "ja": bool(my_vote)})
                alli.events.append(rec)

    def judge_alliance(self, state: SHState, action: Action,
                       actor: int) -> List[dict]:
        """Judge gov-pact honour/betray at enactment.

        For an active ``gov_pact`` whose members include the Chancellor
        (``actor``), the terms may pin a promised policy party via
        ``{"policy": "Liberal"|"Fascist"}``. Enacting the promised party =
        honored; the opposite = betrayed. If no policy is pinned, the gov-pact
        is treated as a cooperation promise and enacting Liberal counts as
        honoured (the cooperative default), Fascist as betrayed.
        """
        events: List[dict] = []
        if action.get("type") != "enact" or actor is None:
            return events
        enacted = action.get("policy")
        alli = state.alli
        for al in alli.alliances.values():
            if al.status != "active" or al.kind != "gov_pact":
                continue
            if actor not in al.members:
                continue
            promised = al.terms.get("policy")
            if isinstance(promised, str):
                honoured = (enacted == promised) or (
                    promised.lower().startswith("l") and enacted == LIBERAL) or (
                    promised.lower().startswith("f") and enacted == FASCIST)
            else:
                honoured = (enacted == LIBERAL)
            event = "honored" if honoured else "betrayed"
            cp = [m for m in al.members if m != actor]
            rec = new_event(event, al, turn=state.turn, actor=actor,
                            counterparty=cp,
                            action_ref={"type": "enact", "policy": enacted})
            alli.events.append(rec)
            events.append(rec)
        return events

    # --------------------------------------------------------- observations
    def observations(self, prev_state, new_state, action: Action,
                     actor: int) -> List[Obs]:
        t = action.get("type")
        if t in ("say", "whisper", "pass_talk",
                 "alliance_propose", "alliance_accept", "alliance_decline",
                 "alliance_break"):
            return self.nego_observations(prev_state, new_state, action, actor)

        state: SHState = new_state

        # Private reveals: Investigate + Policy Peek reach ONLY the president.
        if t == "investigate":
            target = int(action.get("target"))
            party = LIBERAL if state.roles[target] == LIBERAL else FASCIST
            pres = actor
            full = {"type": "reveal", "what": "investigate_loyalty",
                    "data": {"president": pres, "target": target, "party": party}}
            # Bystanders learn only that an investigation happened.
            public = {"type": "action", "player": actor,
                      "action": {"type": "investigate", "target": target}}
            obs_list = [Obs(audience=[pres], payload=full, log=full)]
            others = [s for s in range(self.n_players) if s != pres]
            if others:
                obs_list.append(Obs(audience=others, payload=public,
                                    log={"type": "action", "player": actor,
                                         "action": {"type": "investigate",
                                                    "target": target},
                                         "private": "party hidden from bystanders"}))
            return obs_list

        if t == "peek_ack":
            pres = actor
            top3 = list(reversed(state.deck[-3:]))
            full = {"type": "reveal", "what": "policy_peek",
                    "data": {"president": pres, "top3": top3}}
            public = {"type": "action", "player": actor,
                      "action": {"type": "policy_peek"}}
            obs_list = [Obs(audience=[pres], payload=full, log=full)]
            others = [s for s in range(self.n_players) if s != pres]
            if others:
                obs_list.append(Obs(audience=others, payload=public))
            return obs_list

        # Everything else is public (votes, nominations, enactments,
        # executions, special elections, veto consents): broadcast to all.
        return [Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

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

    # --------------------------------------------------- watcher hooks
    def god_view(self, state: SHState) -> dict:
        return {
            "roles": list(state.roles),
            "deck_top3": list(reversed(state.deck[-3:])),
            "deck_size": len(state.deck),
            "discard_size": len(state.discard),
            "private_info": {str(k): list(v)
                             for k, v in state.private_info.items()},
        }

    def snapshot(self, state: SHState) -> dict:
        public = {
            "phase": state.phase,
            "n_players": state.n_players,
            "alive": list(state.alive),
            "enacted_liberal": state.enacted_liberal,
            "enacted_fascist": state.enacted_fascist,
            "election_tracker": state.election_tracker,
            "president_idx": state.president_idx,
            "chancellor_idx": state.chancellor_idx,
            "last_president": state.last_president,
            "last_chancellor": state.last_chancellor,
            "veto_unlocked": state.enacted_fascist >= VETO_UNLOCK_FASCIST,
            "winner_team": state.winner_team,
            "win_reason": state.win_reason,
        }
        hidden = self.god_view(state)
        return {"public": public, "hidden": hidden}

    def render_board(self, state: SHState, *, reveal: str = "god") -> str:
        lib = "[L]" * state.enacted_liberal + "[ ]" * (5 - state.enacted_liberal)
        fas = "[F]" * state.enacted_fascist + "[ ]" * (6 - state.enacted_fascist)
        lines = [
            f"=== Secret Hitler ({state.n_players}p) - phase={state.phase} ===",
            f"Liberal track : {lib}  ({state.enacted_liberal}/5)",
            f"Fascist track : {fas}  ({state.enacted_fascist}/6)",
            f"Election tracker: {'X' * state.election_tracker}"
            f"{'.' * (3 - state.election_tracker)}  ({state.election_tracker}/3)",
            f"President: P{state.president_idx}   "
            f"Chancellor: {('P' + str(state.chancellor_idx)) if state.chancellor_idx is not None else '-'}",
        ]
        seat_bits = []
        for i in range(state.n_players):
            alive = "alive" if state.alive[i] else "DEAD"
            if reveal == "god":
                seat_bits.append(f"P{i}={state.roles[i]}({alive})")
            else:
                seat_bits.append(f"P{i}({alive})")
        lines.append("Seats: " + ", ".join(seat_bits))
        if state.winner_team:
            lines.append(f"WINNER: {state.winner_team} - {state.win_reason}")
        return "\n".join(lines)
