"""One Night Werewolf (3-10 players, full canonical role set).

This is the v2 feature-parity implementation (spec §5.1). It composes the
shared ``MessagingMixin`` + ``AllianceMixin`` so the DAY phase is a full
negotiation sub-phase (public ``say`` + private ``whisper`` + vote-bloc/truce
alliances), and the VOTE phase is pact-instrumented (a vote against an active
ally = betrayed; voting as agreed = honored).

Deck
----
The deck dealt for an n-player match is always ``n + 3`` cards (n to seats + 3
to the center). The default role list per player count comes from
``maps/onw_roles.ROLE_COUNTS_BY_N``; a caller may override it with the
``roles`` constructor argument (a list of role strings of length ``n + 3``).

Roles
-----
Werewolf, Minion, Mason (x2), Seer, Robber, Troublemaker, Insomniac, Hunter,
Tanner, Drunk, Doppelganger, Villager.

Canonical wake order (``maps/onw_roles.WAKE_ORDER``):
  Doppelganger -> Werewolf -> Minion -> Mason -> Seer -> Robber ->
  Troublemaker -> Drunk -> Insomniac.
Hunter, Tanner, and Villager have no night action.

Role canon
----------
- Werewolves see each other; a lone wolf may peek at one center card.
- Minion wakes after the wolves, learns who the wolves are, wins WITH the
  wolves, and is death-safe (the wolves do not need the Minion to survive).
- Masons wake together and learn each other (or learn they are the lone Mason).
- Seer inspects one player's card OR two center cards.
- Robber swaps its card with a target's and then learns its new card.
- Troublemaker swaps two other players' cards without looking.
- Drunk swaps its card with an (unseen) center card — it does NOT learn the
  new role, so it may be a werewolf and not know it.
- Insomniac wakes last and sees its own final (post-swap) card.
- Doppelganger wakes first, copies a target's CURRENT role, and — if that role
  is a night-acting role — a dynamic second ``night_queue`` entry lets the
  Doppelganger act in that role's slot (e.g. Doppelganger-Seer peeks, a
  Doppelganger-Robber swaps, Doppelganger-Werewolf joins the wolf team, etc.).
  Doppelganger-Hunter shoots on death like a Hunter.

Win / reward edge cases (spec §5.1)
-----------------------------------
- Village team wins iff at least one werewolf-TEAM member (a Werewolf card OR
  a Doppelganger-Werewolf) is eliminated.
- Werewolf team (Werewolves + Minion + Doppelganger-Werewolf) wins iff no
  Werewolf card dies (and at least one werewolf is in play).
- Tanner is independent and wins iff the Tanner *themself* dies; if the Tanner
  dies, the Tanner wins and the village does NOT (even if a wolf also died).
- Hunter (and Doppelganger-Hunter) on death eliminates their vote target,
  which can in turn kill a werewolf.
- No-werewolf edge case preserved: if no Werewolf is in play after swaps, the
  village wins iff nobody dies, else "nobody" wins.

Documented simplification
--------------------------
- Doppelganger-Drunk re-swap nuance: a Doppelganger who copies the *Drunk* role
  performs the Drunk's blind center swap, but the rare cascade where that swap
  interacts with the original Drunk's own later swap is resolved in plain
  queue order rather than with the optional tournament re-swap nuance. This is
  the only allowed §5.1 cut and is surfaced in ``config`` via the module-level
  ``DOPPELGANGER_DRUNK_RESWAP`` flag (False == simplified).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base import Action, Game, Obs, ParseError
from ..config import GameConfig
from ..messaging import MessagingMixin, NegotiationState
from ..alliances import AllianceMixin, AllianceState, new_event
from ..maps.onw_roles import (
    ALL_ROLES,
    ROLE_COUNTS_BY_N,
    TEAM,
    WAKE_ORDER,
)

# ---------------------------------------------------------------- role names
WEREWOLF = "Werewolf"
MINION = "Minion"
MASON = "Mason"
SEER = "Seer"
ROBBER = "Robber"
TROUBLEMAKER = "Troublemaker"
INSOMNIAC = "Insomniac"
HUNTER = "Hunter"
TANNER = "Tanner"
DRUNK = "Drunk"
DOPPELGANGER = "Doppelganger"
VILLAGER = "Villager"

# Roles that have a "real" night action slot (Mason/Werewolf/Minion only peek
# at allies; the others mutate or inspect). Used to decide whether a copied
# Doppelganger role needs a second night_queue entry.
_NIGHT_ROLES = set(WAKE_ORDER)

# ----------------------------------------------------------------- phases
PHASE_NIGHT = "night"
PHASE_DAY = "day"          # == the negotiation sub-phase tag
PHASE_VOTE = "vote"
PHASE_TERMINAL = "terminal"

# Documented Doppelganger-Drunk simplification flag (spec §5.1).
DOPPELGANGER_DRUNK_RESWAP = False


@dataclass
class ONWState:
    """Game state for One Night Werewolf (v2).

    Both the canonical initial roles and the current (post-swap) roles are
    tracked. Hidden-info masking in ``render_prompt`` / ``observations`` ensures
    each seat only sees what its role permits.
    """

    n_players: int = 5
    initial_roles: List[str] = field(default_factory=list)    # length n_players
    center_initial: List[str] = field(default_factory=list)   # length 3
    current_roles: List[str] = field(default_factory=list)    # post-swap
    center_current: List[str] = field(default_factory=list)   # post-swap

    # Doppelganger bookkeeping: seat -> copied role (if any).
    doppel_copy: Dict[int, str] = field(default_factory=dict)

    # Night action queue: list of (player_idx, role) pairs in wake order. A
    # Doppelganger may insert a dynamic second (seat, copied_role) entry.
    night_queue: List[Tuple[int, str]] = field(default_factory=list)
    night_results: Dict[int, str] = field(default_factory=dict)  # own action result text

    # Vote phase.
    vote_queue: List[int] = field(default_factory=list)
    votes: Dict[int, int] = field(default_factory=dict)        # voter -> target

    # Phase tracking.
    phase: str = PHASE_NIGHT

    # Negotiation + alliance sub-state (spec integration recipe).
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)
    turn: int = 0

    # Outcome.
    eliminated: List[int] = field(default_factory=list)
    winner_team: Optional[str] = None     # "village" | "werewolves" | "tanner" | "nobody"
    win_reason: str = ""

    # Internal guard so honour/betrayal is judged exactly once at resolution.
    _alliances_judged: bool = False


class OneNightWerewolf(MessagingMixin, AllianceMixin, Game):
    """One Night Werewolf (3-10p, full role set). See module docstring."""

    name = "one_night_werewolf"
    n_players = 5

    def __init__(self, config: "Optional[GameConfig]" = None,
                 roles: "Optional[List[str]]" = None,
                 n_players: "Optional[int]" = None) -> None:
        """Construct a match.

        Parameters
        ----------
        config : GameConfig | None
            Messaging/alliance/observability config. Defaults to messaging +
            alliances ON.
        roles : list[str] | None
            Explicit deck (length must equal ``n_players + 3``). If None, the
            balanced default from ``ROLE_COUNTS_BY_N[n_players]`` is used.
        n_players : int | None
            Number of seats (3-10). Defaults to the class default (5) so the
            no-arg constructor keeps working (INV-4).
        """
        super().__init__(config)
        if n_players is not None:
            if not (3 <= n_players <= 10):
                raise ValueError("One Night Werewolf supports 3-10 players")
            self.n_players = int(n_players)
        if roles is not None:
            expected = self.n_players + 3
            if len(roles) != expected:
                raise ValueError(
                    f"roles list must have length n_players+3 = {expected}, "
                    f"got {len(roles)}"
                )
            for r in roles:
                if r not in ALL_ROLES:
                    raise ValueError(f"unknown role: {r!r}")
            self._roles = list(roles)
        else:
            self._roles = list(ROLE_COUNTS_BY_N[self.n_players])

    # ------------------------------------------------------------------ setup
    def initial_state(self, rng) -> ONWState:
        deck = list(self._roles)
        rng.shuffle(deck)
        initial = deck[: self.n_players]
        center = deck[self.n_players: self.n_players + 3]

        st = ONWState(
            n_players=self.n_players,
            initial_roles=list(initial),
            center_initial=list(center),
            current_roles=list(initial),
            center_current=list(center),
            phase=PHASE_NIGHT,
        )
        st.night_queue = self._build_night_queue(initial)
        st.vote_queue = list(range(self.n_players))
        return st

    @staticmethod
    def _build_night_queue(initial: List[str]) -> List[Tuple[int, str]]:
        """Build the canonical (seat, role) wake queue from the initial deal."""
        nq: List[Tuple[int, str]] = []
        for role in WAKE_ORDER:
            for seat, r in enumerate(initial):
                if r == role:
                    nq.append((seat, role))
        return nq

    # ----------------------------------------------------------- active_player
    def active_player(self, state: ONWState) -> int:
        if state.phase == PHASE_NIGHT:
            if state.night_queue:
                return state.night_queue[0][0]
            return -1
        if state.phase == PHASE_DAY:
            # DAY is the negotiation sub-phase.
            return self.nego_active_player(state)
        if state.phase == PHASE_VOTE:
            if state.vote_queue:
                return state.vote_queue[0]
            return -1
        return -1

    # --------------------------------------------------------- living seats
    def living_seats(self, state: ONWState) -> List[int]:
        """All seats are "alive" during night/day/vote in One Night Werewolf
        (deaths happen only at resolution). Eliminations matter post-terminal
        only, so every seat may speak / be addressed throughout."""
        return list(range(state.n_players))

    # -------------------------------------------------------------- legal_actions
    def legal_actions(self, state: ONWState, player: int) -> List[Action]:
        if state.phase == PHASE_NIGHT and state.night_queue:
            return self._night_legal_actions(state, player)
        if state.phase == PHASE_DAY:
            return self.nego_legal_actions(state, player)
        if state.phase == PHASE_VOTE:
            return [{"type": "vote", "target": t}
                    for t in range(self.n_players) if t != player]
        return []

    def _night_legal_actions(self, state: ONWState, player: int) -> List[Action]:
        seat, role = state.night_queue[0]
        if seat != player:
            return []
        if role == DOPPELGANGER:
            return ([{"type": "doppel_copy", "target": t}
                     for t in range(self.n_players) if t != player])
        if role == WEREWOLF:
            others = self._wolf_seats(state, exclude=player)
            if others:
                return [{"type": "wolf_acknowledge"}]
            return ([{"type": "wolf_peek_center", "center": c} for c in (0, 1, 2)]
                    + [{"type": "wolf_no_peek"}])
        if role == MINION:
            return [{"type": "minion_acknowledge"}]
        if role == MASON:
            return [{"type": "mason_acknowledge"}]
        if role == SEER:
            actions: List[Action] = []
            actions += [{"type": "seer_peek_player", "target": t}
                        for t in range(self.n_players) if t != player]
            actions += [{"type": "seer_peek_center", "pair": list(p)}
                        for p in ((0, 1), (0, 2), (1, 2))]
            actions += [{"type": "seer_pass"}]
            return actions
        if role == ROBBER:
            return ([{"type": "robber_swap", "target": t}
                     for t in range(self.n_players) if t != player]
                    + [{"type": "robber_pass"}])
        if role == TROUBLEMAKER:
            actions = []
            others = [i for i in range(self.n_players) if i != player]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    actions.append({"type": "tm_swap", "a": others[i], "b": others[j]})
            actions.append({"type": "tm_pass"})
            return actions
        if role == DRUNK:
            return [{"type": "drunk_swap", "center": c} for c in (0, 1, 2)]
        if role == INSOMNIAC:
            return [{"type": "insomniac_check"}]
        return [{"type": "no_action"}]

    def _wolf_seats(self, state: ONWState, *, exclude: "Optional[int]" = None) -> List[int]:
        """Seats whose INITIAL role makes them a wolf for the night-info phase.

        A Doppelganger who copied Werewolf joins the wolf team for the purposes
        of the wolves recognising one another.
        """
        out = []
        for i, r in enumerate(state.initial_roles):
            is_wolf = (r == WEREWOLF) or (state.doppel_copy.get(i) == WEREWOLF)
            if is_wolf and i != exclude:
                out.append(i)
        return out

    # --------------------------------------------------------------- rendering
    def render_prompt(self, state: ONWState, player: int) -> str:
        public = self._public_header(state)
        private = self._private_header(state, player)

        if state.phase == PHASE_NIGHT and state.night_queue:
            seat, role = state.night_queue[0]
            if seat != player:
                return f"{public}\n\n{private}\n\nIt is not your turn yet."
            return f"{public}\n\n{private}\n\n{self._night_prompt(state, player, role)}"
        if state.phase == PHASE_DAY:
            return f"{public}\n\n{private}\n\n{self._day_prompt(state, player)}"
        if state.phase == PHASE_VOTE:
            return f"{public}\n\n{private}\n\n{self._vote_prompt(state, player)}"
        return public + "\n\n[Game over.]"

    def _public_header(self, state: ONWState) -> str:
        counts: Dict[str, int] = {}
        for r in self._roles:
            counts[r] = counts.get(r, 0) + 1
        deck_summary = ", ".join(f"{r}x{counts[r]}" for r in sorted(counts))
        return (
            f"=== One Night Werewolf ===\n"
            f"Players: {state.n_players} (numbered 0..{state.n_players-1}).\n"
            f"Deck ({len(self._roles)} cards, {state.n_players} dealt + 3 in center): "
            f"{deck_summary}\n"
            f"Phase: {state.phase.upper()}.\n"
        )

    def _private_header(self, state: ONWState, player: int) -> str:
        my_role = state.initial_roles[player]
        bits = [f"You are player {player}. Your starting role is {my_role}."]
        copied = state.doppel_copy.get(player)
        if copied:
            bits.append(f"As the Doppelganger you copied the {copied} role.")
        info = state.night_results.get(player)
        if info:
            bits.append(f"From your night action: {info}")
        # Masked negotiation transcript (the single masking point for chat).
        log = self.render_message_log(state, player)
        if log:
            bits.append(f"Discussion so far:\n{log}")
        return "\n".join(bits)

    def _night_prompt(self, state: ONWState, player: int, role: str) -> str:
        if role == DOPPELGANGER:
            return (
                "Night action - Doppelganger:\n"
                "Choose ONE other player whose role you copy. If that role acts at\n"
                "night, you will then act in that role's slot.\n"
                "Respond with exactly:\n"
                "<action>copy X</action>   (X = a player index other than you)"
            )
        if role == WEREWOLF:
            others = self._wolf_seats(state, exclude=player)
            if others:
                return (
                    "Night action - Werewolf:\n"
                    f"The other werewolf(s) are players {others}.\n"
                    "Respond with exactly:\n"
                    "<action>acknowledge</action>"
                )
            return (
                "Night action - Lone Werewolf:\n"
                "You may peek at ONE center card (0, 1, or 2), or skip.\n"
                "Respond with exactly one of:\n"
                "<action>peek_center 0</action>\n"
                "<action>peek_center 1</action>\n"
                "<action>peek_center 2</action>\n"
                "<action>skip</action>"
            )
        if role == MINION:
            return (
                "Night action - Minion:\n"
                "You wake to learn who the werewolves are. You win WITH the wolves,\n"
                "and you are death-safe (the wolves need only avoid losing a wolf).\n"
                "Respond with exactly:\n"
                "<action>acknowledge</action>"
            )
        if role == MASON:
            return (
                "Night action - Mason:\n"
                "You wake with the other Mason (if any) and learn each other.\n"
                "Respond with exactly:\n"
                "<action>acknowledge</action>"
            )
        if role == SEER:
            return (
                "Night action - Seer:\n"
                "Inspect ONE other player's card, OR TWO center cards, OR pass.\n"
                "Respond with exactly one of (X = a player index other than you;\n"
                "center pairs are {0,1}, {0,2}, {1,2}):\n"
                "<action>peek_player X</action>\n"
                "<action>peek_center 0 1</action>\n"
                "<action>peek_center 0 2</action>\n"
                "<action>peek_center 1 2</action>\n"
                "<action>pass</action>"
            )
        if role == ROBBER:
            return (
                "Night action - Robber:\n"
                "Swap your card with one other player's and look at your new card, OR pass.\n"
                "Respond with exactly one of:\n"
                "<action>swap X</action>   (X = a player index, not you)\n"
                "<action>pass</action>"
            )
        if role == TROUBLEMAKER:
            return (
                "Night action - Troublemaker:\n"
                "Swap two OTHER players' cards without looking, OR pass.\n"
                "Respond with exactly one of:\n"
                "<action>swap X Y</action>   (X, Y distinct, neither is you)\n"
                "<action>pass</action>"
            )
        if role == DRUNK:
            return (
                "Night action - Drunk:\n"
                "You MUST swap your card with a center card (0, 1, or 2) WITHOUT\n"
                "looking - you will not learn your new role.\n"
                "Respond with exactly one of:\n"
                "<action>swap_center 0</action>\n"
                "<action>swap_center 1</action>\n"
                "<action>swap_center 2</action>"
            )
        if role == INSOMNIAC:
            return (
                "Night action - Insomniac:\n"
                "You wake last and look at your own (possibly swapped) card.\n"
                "Respond with exactly:\n"
                "<action>check</action>"
            )
        return "Night action - no action available. Respond with <action>none</action>."

    def _day_prompt(self, state: ONWState, player: int) -> str:
        nego = state.nego
        budget = nego.budget.get(player, 0)
        lines = [
            "Day phase (open discussion + alliances). It is your turn to speak.",
            f"Round {nego.round_idx + 1} of {nego.max_rounds}. "
            f"You have {budget} message(s) left this slot.",
            "You may send a PUBLIC message, a PRIVATE whisper, or pass; and you",
            "may propose / accept / decline / break a vote-bloc or truce alliance.",
            "Respond with exactly ONE of:",
            "<say>your public message</say>",
            "<whisper to=2,4>your private message</whisper>",
            "<pass></pass>",
            "<ally propose to=2,3 kind=vote_bloc>let us both vote player 5</ally>",
            "<ally accept 7>   <ally decline 7>   <ally break 7>reason</ally>",
        ]
        return "\n".join(lines)

    def _vote_prompt(self, state: ONWState, player: int) -> str:
        targets = [i for i in range(self.n_players) if i != player]
        return (
            "Vote phase. You MUST choose exactly ONE other player to eliminate. "
            "You CANNOT pass, abstain, or skip the vote. If you agreed a vote-bloc, "
            "voting as agreed HONORS the pact; voting against an ally BETRAYS it.\n"
            f"Valid choices: {targets}.\n"
            "End your reply with the tag on its own line, e.g. <vote>"
            f"{targets[0]}</vote>. You may reason first, but you must include the tag."
        )

    # --------------------------------------------------------------- parsing
    _ACT_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
    _COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    _VOTE_RE = re.compile(r"<vote>\s*(\d+)\s*</vote>", re.IGNORECASE)
    # Lenient fallbacks: "vote 3", "vote for player 3", "I vote P3", "eliminate 3".
    _VOTE_LOOSE_RE = re.compile(
        r"(?:vote(?:\s+for)?|eliminate|lynch)\s*(?:player\s*|seat\s*|p)?\s*#?\s*(\d+)",
        re.IGNORECASE)

    def parse_action(self, state: ONWState, player: int, text: str) -> Action:
        if state.phase == PHASE_NIGHT and state.night_queue:
            return self._parse_night(state, player, text)
        if state.phase == PHASE_DAY:
            return self.nego_parse(state, player, text)
        if state.phase == PHASE_VOTE:
            # Strip HTML-comment reasoning blocks the model sometimes appends.
            clean = self._COMMENT_RE.sub(" ", text)
            if "<pass" in clean.lower() and not self._VOTE_RE.search(clean):
                raise ParseError(
                    "you cannot pass or abstain in the vote phase — you must "
                    "name exactly one other player as <vote>X</vote>")
            m = self._VOTE_RE.search(clean) or self._VOTE_LOOSE_RE.search(clean)
            if not m:
                # Last resort: a lone integer that is a valid target.
                for tok in re.findall(r"\d+", clean):
                    v = int(tok)
                    if v != player and 0 <= v < self.n_players:
                        m = None
                        target = v
                        break
                else:
                    raise ParseError("expected a vote tag containing a player number, e.g. <vote>1</vote>")
            else:
                target = int(m.group(1))
            if target == player:
                raise ParseError("you cannot vote for yourself")
            if not (0 <= target < self.n_players):
                raise ParseError(f"vote target {target} out of range")
            return {"type": "vote", "target": target}
        raise ParseError(f"no action expected in phase {state.phase}")

    def _parse_night(self, state: ONWState, player: int, text: str) -> Action:
        seat, role = state.night_queue[0]
        if seat != player:
            raise ParseError("not your night-action turn")
        m = self._ACT_RE.search(text)
        if not m:
            raise ParseError("expected <action>...</action>")
        body = m.group(1).strip().lower()

        if role == DOPPELGANGER:
            md = re.match(r"copy\s+(\d+)$", body)
            if md:
                t = int(md.group(1))
                if t == player or not (0 <= t < self.n_players):
                    raise ParseError("invalid copy target")
                return {"type": "doppel_copy", "target": t}
            raise ParseError("expected copy X")
        if role == WEREWOLF:
            others = self._wolf_seats(state, exclude=player)
            if others:
                if body != "acknowledge":
                    raise ParseError("werewolf must respond <action>acknowledge</action>")
                return {"type": "wolf_acknowledge"}
            if body == "skip":
                return {"type": "wolf_no_peek"}
            mc = re.match(r"peek_center\s+([012])$", body)
            if mc:
                return {"type": "wolf_peek_center", "center": int(mc.group(1))}
            raise ParseError("expected peek_center N or skip")
        if role == MINION:
            if body != "acknowledge":
                raise ParseError("minion must respond <action>acknowledge</action>")
            return {"type": "minion_acknowledge"}
        if role == MASON:
            if body != "acknowledge":
                raise ParseError("mason must respond <action>acknowledge</action>")
            return {"type": "mason_acknowledge"}
        if role == SEER:
            if body == "pass":
                return {"type": "seer_pass"}
            mp = re.match(r"peek_player\s+(\d+)$", body)
            if mp:
                t = int(mp.group(1))
                if t == player or not (0 <= t < self.n_players):
                    raise ParseError("invalid peek_player target")
                return {"type": "seer_peek_player", "target": t}
            mc = re.match(r"peek_center\s+([012])\s+([012])$", body)
            if mc:
                a, b = int(mc.group(1)), int(mc.group(2))
                if a == b:
                    raise ParseError("peek_center indices must differ")
                return {"type": "seer_peek_center", "pair": sorted((a, b))}
            raise ParseError("expected peek_player X, peek_center A B, or pass")
        if role == ROBBER:
            if body == "pass":
                return {"type": "robber_pass"}
            mr = re.match(r"swap\s+(\d+)$", body)
            if mr:
                t = int(mr.group(1))
                if t == player or not (0 <= t < self.n_players):
                    raise ParseError("invalid swap target")
                return {"type": "robber_swap", "target": t}
            raise ParseError("expected swap X or pass")
        if role == TROUBLEMAKER:
            if body == "pass":
                return {"type": "tm_pass"}
            mt = re.match(r"swap\s+(\d+)\s+(\d+)$", body)
            if mt:
                a, b = int(mt.group(1)), int(mt.group(2))
                if a == b or a == player or b == player:
                    raise ParseError("swap targets must differ from each other and from you")
                if not (0 <= a < self.n_players) or not (0 <= b < self.n_players):
                    raise ParseError("swap indices out of range")
                return {"type": "tm_swap", "a": a, "b": b}
            raise ParseError("expected swap X Y or pass")
        if role == DRUNK:
            mc = re.match(r"swap_center\s+([012])$", body)
            if mc:
                return {"type": "drunk_swap", "center": int(mc.group(1))}
            raise ParseError("expected swap_center N")
        if role == INSOMNIAC:
            if body != "check":
                raise ParseError("insomniac must respond <action>check</action>")
            return {"type": "insomniac_check"}
        return {"type": "no_action"}

    # --------------------------------------------------------------- step
    def step(self, state: ONWState, action: Action) -> ONWState:
        t = action.get("type")

        # Negotiation / alliance actions are routed to the shared mixin.
        if t in ("say", "whisper", "pass_talk", "alliance_propose",
                 "alliance_accept", "alliance_decline", "alliance_break"):
            state.turn += 1
            return self.nego_step(state, action)

        if t == "advance_phase":
            return self._advance_phase(state)

        if state.phase == PHASE_NIGHT and state.night_queue:
            return self._step_night(state, action)

        if state.phase == PHASE_VOTE and t == "vote":
            state.turn += 1
            voter = state.vote_queue[0]
            state.votes[voter] = int(action["target"])
            state.vote_queue = state.vote_queue[1:]
            if not state.vote_queue:
                state = self._resolve(state)
            return state

        return state

    def _step_night(self, state: ONWState, action: Action) -> ONWState:
        state.turn += 1
        seat, role = state.night_queue[0]
        t = action.get("type")
        new = state

        if t == "doppel_copy":
            target = action["target"]
            copied = new.current_roles[target]
            new.doppel_copy[seat] = copied
            new.night_results[seat] = (
                f"You copied player {target}'s role: {copied}."
            )
            # If the copied role acts at night, insert a dynamic second entry
            # so the Doppelganger acts in that role's slot. We insert it just
            # after the current Doppelganger entry is popped (i.e. at the front
            # of the remaining queue is wrong — it must respect wake order),
            # so we place it at the canonical position for that role among the
            # entries still to come.
            new.night_queue = new.night_queue[1:]
            if copied in _NIGHT_ROLES:
                self._insert_doppel_night_slot(new, seat, copied)
            return self._after_night_action(new)

        if t == "wolf_acknowledge":
            others = self._wolf_seats(new, exclude=seat)
            new.night_results[seat] = f"You confirmed the OTHER werewolves are players {others}."
        elif t == "wolf_no_peek":
            new.night_results[seat] = "You chose not to peek at any center card."
        elif t == "wolf_peek_center":
            c = action["center"]
            new.night_results[seat] = f"Center card {c} is {new.center_current[c]}."
        elif t == "minion_acknowledge":
            wolves = self._wolf_seats(new)
            if wolves:
                new.night_results[seat] = (
                    f"You are the Minion. The werewolves are players {wolves}. "
                    "You win with them."
                )
            else:
                new.night_results[seat] = (
                    "You are the Minion. There are no werewolves among the players "
                    "(both are in the center). You win if no player is killed... "
                    "as a wolf-team member."
                )
        elif t == "mason_acknowledge":
            masons = [i for i, r in enumerate(new.initial_roles)
                      if r == MASON or new.doppel_copy.get(i) == MASON]
            others = [m for m in masons if m != seat]
            if others:
                new.night_results[seat] = f"The other Mason(s) are players {others}."
            else:
                new.night_results[seat] = (
                    "You are the only Mason among the players (the other Mason "
                    "card is in the center)."
                )
        elif t == "seer_pass":
            new.night_results[seat] = "You passed."
        elif t == "seer_peek_player":
            target = action["target"]
            new.night_results[seat] = (
                f"Player {target}'s card is {new.current_roles[target]}."
            )
        elif t == "seer_peek_center":
            a, b = action["pair"]
            new.night_results[seat] = (
                f"Center {a} = {new.center_current[a]}; "
                f"Center {b} = {new.center_current[b]}."
            )
        elif t == "robber_pass":
            new.night_results[seat] = "You passed."
        elif t == "robber_swap":
            target = action["target"]
            mine = new.current_roles[seat]
            theirs = new.current_roles[target]
            new.current_roles[seat] = theirs
            new.current_roles[target] = mine
            new.night_results[seat] = (
                f"You swapped with player {target}. Your new role is {theirs}."
            )
        elif t == "tm_pass":
            new.night_results[seat] = "You passed."
        elif t == "tm_swap":
            a = action["a"]; b = action["b"]
            ra = new.current_roles[a]; rb = new.current_roles[b]
            new.current_roles[a] = rb; new.current_roles[b] = ra
            new.night_results[seat] = f"You swapped players {a} and {b} (no peek)."
        elif t == "drunk_swap":
            c = action["center"]
            mine = new.current_roles[seat]
            cc = new.center_current[c]
            new.current_roles[seat] = cc
            new.center_current[c] = mine
            new.night_results[seat] = (
                f"You swapped your card with center card {c} (blind - you do "
                "not learn your new role)."
            )
        elif t == "insomniac_check":
            new.night_results[seat] = (
                f"You wake and see your final card: {new.current_roles[seat]}."
            )
        else:
            new.night_results[seat] = "(no night action)"

        new.night_queue = new.night_queue[1:]
        return self._after_night_action(new)

    def _insert_doppel_night_slot(self, state: ONWState, seat: int, copied: str) -> None:
        """Insert a dynamic (seat, copied) entry into the night queue at the
        canonical wake position for ``copied`` among the entries still to come.

        Because the Doppelganger wakes FIRST, every other role's slot is still
        ahead in the queue, so the copied slot is inserted just before the
        first existing entry whose role wakes no earlier than ``copied``.
        """
        rank = WAKE_ORDER.index(copied)
        insert_at = len(state.night_queue)
        for idx, (_, r) in enumerate(state.night_queue):
            if WAKE_ORDER.index(r) >= rank:
                insert_at = idx
                break
        state.night_queue.insert(insert_at, (seat, copied))

    def _after_night_action(self, state: ONWState) -> ONWState:
        """If the night queue is now empty, open the DAY negotiation phase."""
        if not state.night_queue:
            self.start_negotiation(state, return_phase=PHASE_VOTE,
                                   rounds=self.config.nego_rounds)
            state.phase = PHASE_DAY
        return state

    def _advance_phase(self, state: ONWState) -> ONWState:
        """Empty-queue safety net so a match can never stall."""
        if state.phase == PHASE_NIGHT and not state.night_queue:
            self.start_negotiation(state, return_phase=PHASE_VOTE,
                                   rounds=self.config.nego_rounds)
            state.phase = PHASE_DAY
        elif state.phase == PHASE_DAY and not state.nego.speak_queue:
            # Negotiation finished (mixin set phase to return_phase) but the
            # runner asked us to advance; just ensure we are in vote.
            if state.phase == PHASE_DAY:
                self._exit_negotiation(state)
        elif state.phase == PHASE_VOTE and not state.vote_queue:
            state = self._resolve(state)
        return state

    # --------------------------------------------------------- observations
    def observations(self, prev_state, new_state, action: Action,
                     actor: int) -> List[Obs]:
        """Route + mask every action.

        Night peeks are emitted as ``reveal`` observations scoped to the peeker
        only; negotiation/alliance actions go through the shared mixin; public
        actions (votes, acknowledgements, advance_phase) fall back to the
        default broadcast so a spectator/log sees the public game flow (and the
        legacy broadcast-obs path still fires for INV-1).
        """
        t = action.get("type")

        if t in ("say", "whisper", "pass_talk") or (
                t and t.startswith("alliance_")):
            return self.nego_observations(prev_state, new_state, action, actor)

        # Night actions that grant the actor private info -> a `reveal` obs
        # scoped to the peeker only (full content in the god-log).
        #
        # tm_swap/drunk_swap/seer_pass/robber_pass/tm_pass/wolf_no_peek used
        # to fall through to the public broadcast below, outing the acting
        # seat's role/choice to every bystander for the rest of the match
        # (audit "ONW night-action broadcast leak", Critical). They carry no
        # LESS actor-identifying info than the other night actions above, so
        # they get the exact same scoping.
        reveal_types = {
            "wolf_acknowledge", "wolf_peek_center", "minion_acknowledge",
            "mason_acknowledge", "seer_peek_player", "seer_peek_center",
            "robber_swap", "insomniac_check", "doppel_copy",
            "tm_swap", "drunk_swap", "seer_pass", "robber_pass", "tm_pass",
            "wolf_no_peek",
        }
        if t in reveal_types:
            data = {"result": new_state.night_results.get(actor, "")}
            payload = {"type": "reveal", "what": t, "data": data}
            # Audience = the peeker only; the god-log carries the same content.
            return [Obs(audience=[actor], payload=payload)]

        # Public, content-free or board-resolution actions -> default broadcast.
        return [Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

    # ----------------------------------------------------------- alliances
    def alliance_legal_actions(self, state: ONWState, player: int) -> List[Action]:
        """ONW alliance kinds: ``vote_bloc`` (agree to vote a common target)
        and ``truce`` (agree not to vote each other). Accept/decline/break of
        any pending/active alliance plus a vote-bloc + truce proposal to each
        other seat."""
        alli = state.alli
        actions: List[Action] = []
        for al in alli.alliances.values():
            if al.status == "proposed" and player in al.pending:
                actions.append({"type": "alliance_accept", "alliance_id": al.id})
                actions.append({"type": "alliance_decline", "alliance_id": al.id})
            if al.status == "active" and player in al.members:
                actions.append({"type": "alliance_break", "alliance_id": al.id,
                                "reason": ""})
        for other in range(self.n_players):
            if other != player:
                actions.append({"type": "alliance_propose", "to": [other],
                                "kind": "truce", "terms": {}})
                # A vote-bloc names a common target (any third living seat).
                for tgt in range(self.n_players):
                    if tgt not in (player, other):
                        actions.append({
                            "type": "alliance_propose", "to": [other],
                            "kind": "vote_bloc", "terms": {"target": tgt},
                        })
                        break
        return actions

    def judge_alliance(self, state: ONWState, action: Action,
                       actor: int) -> List[dict]:
        """Judge honour/betrayal of active alliances against the final votes.

        Called once at resolution (from ``_resolve``). For each active
        alliance:
          * ``vote_bloc``: a member who voted the agreed target HONORED the
            pact; a member who voted a *fellow member* BETRAYED it. (Voting a
            non-target, non-member seat is neither.)
          * ``truce``: a member who voted a fellow member BETRAYED the truce;
            members who avoided voting each other HONORED it.
        Emits ``alliance_event`` records (honored/betrayed) into ``state.alli``.
        """
        alli = state.alli
        turn = state.turn
        events: List[dict] = []
        for al in alli.alliances.values():
            if al.status not in ("active", "broken"):
                continue
            members = set(al.members)
            if al.kind == "vote_bloc":
                target = al.terms.get("target")
                for m in al.members:
                    vote = state.votes.get(m)
                    if vote is None:
                        continue
                    if vote in members and vote != m:
                        # Voted a fellow bloc member -> betrayal.
                        ev = new_event("betrayed", al, turn=turn, actor=m,
                                       counterparty=[vote],
                                       action_ref={"type": "vote", "target": vote})
                        events.append(ev)
                    elif target is not None and vote == target:
                        ev = new_event("honored", al, turn=turn, actor=m,
                                       counterparty=sorted(members - {m}),
                                       action_ref={"type": "vote", "target": vote})
                        events.append(ev)
            elif al.kind == "truce":
                for m in al.members:
                    vote = state.votes.get(m)
                    if vote is None:
                        continue
                    if vote in members and vote != m:
                        ev = new_event("betrayed", al, turn=turn, actor=m,
                                       counterparty=[vote],
                                       action_ref={"type": "vote", "target": vote})
                        events.append(ev)
                    else:
                        ev = new_event("honored", al, turn=turn, actor=m,
                                       counterparty=sorted(members - {m}),
                                       action_ref={"type": "vote", "target": vote})
                        events.append(ev)
        alli.events.extend(events)
        return events

    # --------------------------------------------------------------- resolve
    def _resolve(self, state: ONWState) -> ONWState:
        # Judge alliance honour/betrayal against the final votes (once).
        if not state._alliances_judged:
            self.judge_alliance(state, {}, -1)
            state._alliances_judged = True

        # Tally votes -> max -> eliminate all tied (with >0 votes).
        tally: Dict[int, int] = {i: 0 for i in range(self.n_players)}
        for v in state.votes.values():
            tally[v] = tally.get(v, 0) + 1
        max_votes = max(tally.values()) if tally else 0
        eliminated = [i for i, c in tally.items() if c == max_votes and c > 0]

        # Hunter (or Doppelganger-Hunter) on death takes their vote target down.
        eliminated_set = set(eliminated)
        for dead in list(eliminated):
            if self._effective_role(state, dead) == HUNTER:
                tgt = state.votes.get(dead)
                if tgt is not None and tgt != dead:
                    eliminated_set.add(tgt)
        eliminated = sorted(eliminated_set)
        state.eliminated = eliminated

        state.winner_team, state.win_reason = self._winner(state, eliminated)
        state.phase = PHASE_TERMINAL
        return state

    def _effective_role(self, state: ONWState, seat: int) -> str:
        """The role a seat counts as for win resolution: its FINAL card, except
        a Doppelganger who copied a role counts as that copy if it still holds
        the Doppelganger card (it didn't get robbed/swapped to something else)."""
        final = state.current_roles[seat]
        if final == DOPPELGANGER and seat in state.doppel_copy:
            return state.doppel_copy[seat]
        return final

    def _is_wolf_card(self, role: str) -> bool:
        return role == WEREWOLF

    def _winner(self, state: ONWState, eliminated: List[int]) -> Tuple[str, str]:
        """Canonical ONW win resolution, returning (winner_team, reason).

        Resolution order (Tanner first, per canon):
          1. If a Tanner died -> Tanner wins (village does NOT).
          2. Else if no Werewolf card is in play among players:
                no one died -> village wins; someone died -> nobody wins.
          3. Else if a Werewolf card died -> village wins.
          4. Else -> werewolves win.
        """
        # 1. Tanner.
        for d in eliminated:
            if self._effective_role(state, d) == TANNER:
                return "tanner", f"Tanner (player {d}) was eliminated."

        wolves_in_play = [i for i in range(self.n_players)
                          if self._effective_role(state, i) == WEREWOLF]
        if not wolves_in_play:
            if not eliminated:
                return "village", "No werewolves in play and nobody was eliminated."
            return "nobody", "No werewolves in play but a villager was lynched."

        wolf_died = any(self._effective_role(state, d) == WEREWOLF
                        for d in eliminated)
        if wolf_died:
            return "village", "At least one werewolf was eliminated."
        return "werewolves", "No werewolf was eliminated."

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: ONWState) -> bool:
        return state.phase == PHASE_TERMINAL

    def _seat_team(self, state: ONWState, seat: int) -> str:
        """Which win-team a seat belongs to (for reward assignment).

        Uses TEAM on the effective role; the Minion is on the werewolf team,
        a Doppelganger-Werewolf is on the werewolf team, a Tanner is its own
        independent team.
        """
        role = self._effective_role(state, seat)
        return TEAM.get(role, "village")

    def rewards(self, state: ONWState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        winner = state.winner_team
        if winner in (None, "nobody"):
            return [0.0] * self.n_players
        out: List[float] = []
        for i in range(self.n_players):
            if winner == "tanner":
                # Only the eliminated Tanner(s) win.
                won = (self._effective_role(state, i) == TANNER
                       and i in state.eliminated)
            elif winner == "werewolves":
                won = self._seat_team(state, i) == "werewolf"
            else:  # village
                won = self._seat_team(state, i) == "village"
            out.append(1.0 if won else 0.0)
        return out

    # ------------------------------------------------------- observability
    def god_view(self, state: ONWState) -> dict:
        """Hidden truth for the watcher: initial + current roles, the center,
        and any Doppelganger copies."""
        return {
            "initial_roles": list(state.initial_roles),
            "current_roles": list(state.current_roles),
            "center_initial": list(state.center_initial),
            "center_current": list(state.center_current),
            "doppel_copy": {str(k): v for k, v in state.doppel_copy.items()},
            "deck": list(self._roles),
        }

    def snapshot(self, state: ONWState) -> dict:
        """Public + hidden board snapshot for ``state_snapshot`` records."""
        public = {
            "phase": state.phase,
            "n_players": state.n_players,
            "votes": {str(k): v for k, v in state.votes.items()},
            "eliminated": list(state.eliminated),
            "winner_team": state.winner_team,
        }
        hidden = {
            "current_roles": list(state.current_roles),
            "center_current": list(state.center_current),
            "doppel_copy": {str(k): v for k, v in state.doppel_copy.items()},
        }
        return {"public": public, "hidden": hidden}

    def render_board(self, state: ONWState, *, reveal: str = "god") -> str:
        """ASCII seat grid. ``reveal`` in {``"god"``, ``"public"``, ``"seatN"``}.

        god      -> every seat's final role shown.
        public   -> roles hidden ("???"), votes/eliminations shown.
        seatN    -> seat N's own role shown, others hidden unless N peeked them
                    (best-effort from N's night_results text is not parsed; we
                    show only N's own role to keep masking trivially correct).
        """
        seat_n = None
        if reveal.startswith("seat"):
            try:
                seat_n = int(reveal[4:])
            except ValueError:
                seat_n = None

        lines = [f"One Night Werewolf - phase {state.phase.upper()}"]
        for i in range(state.n_players):
            if reveal == "god":
                role = state.current_roles[i]
            elif seat_n is not None and i == seat_n:
                role = state.current_roles[i]
            else:
                role = "???"
            tags = []
            if i in state.eliminated:
                tags.append("X-DEAD")
            if i in state.votes:
                tags.append(f"voted->P{state.votes[i]}")
            suffix = ("  " + " ".join(tags)) if tags else ""
            lines.append(f"  P{i}: {role}{suffix}")
        if reveal == "god":
            lines.append("  Center: " + ", ".join(
                f"[{j}]={c}" for j, c in enumerate(state.center_current)))
        if state.winner_team:
            lines.append(f"  WINNER: {state.winner_team}")
        return "\n".join(lines)
