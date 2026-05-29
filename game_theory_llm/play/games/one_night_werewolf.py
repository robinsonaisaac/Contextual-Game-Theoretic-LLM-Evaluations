"""One Night Werewolf (5-player canonical config).

Deck (8 cards total):
  2 Werewolf, 1 Seer, 1 Robber, 1 Troublemaker, 3 Villager
5 dealt to players, 3 to the center.

Phases:
  1. NIGHT (sequential, private actions):
       - Werewolves: see each other; if lone wolf, may peek at 1 center card.
       - Seer: optionally inspect 1 other player's card OR 2 center cards.
       - Robber: optionally swap card with another player, peek at new role.
       - Troublemaker: optionally swap two other players' cards (no peek).
       - Villager: no action.
  2. DAY (free-form discussion, 2 speaking slots per player, round-robin).
  3. VOTE (each player names a target).
  4. RESOLUTION:
       - The player(s) with the most votes die. Ties: all tied die.
       - Village wins iff at least one Werewolf (post-swap role) dies.
       - Edge case: if there are NO Werewolves in play after swaps AND no
         player dies, Village wins automatically.

The state stores both INITIAL_ROLES and CURRENT_ROLES (post-night swaps).
Win conditions use CURRENT_ROLES. Players only see what their role
ALLOWS them to see (their starting role, plus any peeks the night action
granted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base import Action, Game, ParseError


# Roles
WEREWOLF = "Werewolf"
SEER = "Seer"
ROBBER = "Robber"
TROUBLEMAKER = "Troublemaker"
VILLAGER = "Villager"

DECK = [WEREWOLF, WEREWOLF, SEER, ROBBER, TROUBLEMAKER, VILLAGER, VILLAGER, VILLAGER]

# Night action order (the role at the top of the list wakes first)
NIGHT_ORDER = [WEREWOLF, SEER, ROBBER, TROUBLEMAKER]

PHASE_NIGHT = "night"
PHASE_DAY = "day"
PHASE_VOTE = "vote"
PHASE_TERMINAL = "terminal"

# How many speaking slots each player gets in the day phase
DAY_SLOTS_PER_PLAYER = 2


@dataclass
class ONWState:
    """Game state for One Night Werewolf.

    Includes both the canonical initial roles and the current (post-swap)
    roles. Hidden-info masking in `render_prompt` ensures the LLM only
    sees what its role permits.
    """
    n_players: int = 5
    initial_roles: List[str] = field(default_factory=list)   # length n_players
    center_initial: List[str] = field(default_factory=list)  # length 3
    current_roles: List[str] = field(default_factory=list)   # post-swap
    center_current: List[str] = field(default_factory=list)  # post-swap

    # Night action queue: list of (player_idx, role) pairs in night order
    night_queue: List[Tuple[int, str]] = field(default_factory=list)
    night_results: Dict[int, str] = field(default_factory=dict)  # per-player visible result of own action

    # Day phase: alternating speakers, 2 slots each
    day_queue: List[int] = field(default_factory=list)     # whose turn to speak
    day_log: List[Tuple[int, str]] = field(default_factory=list)  # (player_idx, message)

    # Vote phase
    vote_queue: List[int] = field(default_factory=list)
    votes: Dict[int, int] = field(default_factory=dict)    # player_idx -> target_idx

    # Phase tracking
    phase: str = PHASE_NIGHT

    # Outcome
    eliminated: List[int] = field(default_factory=list)
    winner_team: Optional[str] = None  # "village" | "werewolves"


class OneNightWerewolf(Game):
    """One Night Werewolf (5p canonical config). See module docstring."""

    name = "one_night_werewolf"
    n_players = 5

    # ------------------------------------------------------------------ setup
    def initial_state(self, rng) -> ONWState:
        deck = list(DECK)
        rng.shuffle(deck)
        initial = deck[:5]
        center = deck[5:]
        # Compute night queue: for each role in NIGHT_ORDER, find seats
        # that hold that role (initial deal).
        nq: List[Tuple[int, str]] = []
        for role in NIGHT_ORDER:
            for seat, r in enumerate(initial):
                if r == role:
                    nq.append((seat, role))
        # Day queue: 2 slots per player round-robin starting from seat 0.
        dq = [seat for _ in range(DAY_SLOTS_PER_PLAYER)
                for seat in range(self.n_players)]
        # Vote queue: each player votes once.
        vq = list(range(self.n_players))
        return ONWState(
            n_players=self.n_players,
            initial_roles=list(initial),
            center_initial=list(center),
            current_roles=list(initial),
            center_current=list(center),
            night_queue=list(nq),
            day_queue=list(dq),
            vote_queue=list(vq),
            phase=PHASE_NIGHT,
        )

    # ----------------------------------------------------------- active_player
    def active_player(self, state: ONWState) -> int:
        if state.phase == PHASE_NIGHT:
            if state.night_queue:
                return state.night_queue[0][0]
            # All night actions done; move to day. We use a -1 to ask the
            # runner to call step with advance_phase.
            return -1
        if state.phase == PHASE_DAY:
            if state.day_queue:
                return state.day_queue[0]
            return -1
        if state.phase == PHASE_VOTE:
            if state.vote_queue:
                return state.vote_queue[0]
            return -1
        return -1

    # -------------------------------------------------------------- legal_actions
    def legal_actions(self, state: ONWState, player: int) -> List[Action]:
        if state.phase == PHASE_NIGHT and state.night_queue:
            seat, role = state.night_queue[0]
            if seat != player:
                return []
            if role == WEREWOLF:
                # Werewolves: identify other wolves; lone wolf may peek at 1 center
                others = [i for i, r in enumerate(state.initial_roles)
                          if r == WEREWOLF and i != player]
                if others:
                    return [{"type": "wolf_acknowledge"}]
                # Lone wolf — choose a center card to peek
                return [{"type": "wolf_peek_center", "center": c} for c in (0, 1, 2)] + [{"type": "wolf_no_peek"}]
            if role == SEER:
                actions: List[Action] = []
                actions += [{"type": "seer_peek_player", "target": t}
                            for t in range(self.n_players) if t != player]
                # Or peek two center cards: choose a pair {0,1}, {0,2}, {1,2}
                actions += [{"type": "seer_peek_center", "pair": p}
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
            return [{"type": "no_action"}]
        if state.phase == PHASE_DAY:
            return [{"type": "speak", "message": "<freeform>"}]
        if state.phase == PHASE_VOTE:
            return [{"type": "vote", "target": t}
                    for t in range(self.n_players) if t != player]
        return []

    # --------------------------------------------------------------- rendering
    def render_prompt(self, state: ONWState, player: int) -> str:
        # All players first see the public game setup.
        public = self._public_header(state)
        # Per-player private knowledge
        private = self._private_header(state, player)

        if state.phase == PHASE_NIGHT and state.night_queue:
            seat, role = state.night_queue[0]
            if seat != player:
                return f"{public}\n\n{private}\n\nIt is not your turn yet."
            return f"{public}\n\n{private}\n\n{self._night_prompt(state, role)}"
        if state.phase == PHASE_DAY:
            day = self._day_prompt(state, player)
            return f"{public}\n\n{private}\n\n{day}"
        if state.phase == PHASE_VOTE:
            v = self._vote_prompt(state, player)
            return f"{public}\n\n{private}\n\n{v}"
        return public + "\n\n[Game over.]"

    def _public_header(self, state: ONWState) -> str:
        deck_summary = ", ".join(f"{c}×{DECK.count(c)}" for c in sorted(set(DECK)))
        return (
            f"=== One Night Werewolf ===\n"
            f"Players: {state.n_players} (numbered 0..{state.n_players-1}).\n"
            f"Deck (8 cards, 5 dealt + 3 in center): {deck_summary}\n"
            f"Phase: {state.phase.upper()}.\n"
        )

    def _private_header(self, state: ONWState, player: int) -> str:
        my_role = state.initial_roles[player]
        bits = [f"You are player {player}. Your starting role is {my_role}."]
        # Reveal any info this player has accumulated during night actions
        info = state.night_results.get(player)
        if info:
            bits.append(f"From your night action: {info}")
        # Day log so far (everyone sees)
        if state.day_log:
            transcript = "\n".join(f"  P{p}: {m}" for p, m in state.day_log)
            bits.append(f"Discussion so far:\n{transcript}")
        return "\n".join(bits)

    @staticmethod
    def _night_prompt(state: ONWState, role: str) -> str:
        if role == WEREWOLF:
            others = [i for i, r in enumerate(state.initial_roles) if r == WEREWOLF]
            if len(others) > 1:
                return (
                    "Night action — Werewolf:\n"
                    "You wake up and may signal acknowledgment. The other werewolf(s) are "
                    f"players {[i for i in others]}. Respond with exactly:\n"
                    "<action>acknowledge</action>"
                )
            return (
                "Night action — Lone Werewolf:\n"
                "You may peek at ONE center card (positions 0, 1, or 2), or skip.\n"
                "Respond with exactly one of:\n"
                "<action>peek_center 0</action>\n"
                "<action>peek_center 1</action>\n"
                "<action>peek_center 2</action>\n"
                "<action>skip</action>"
            )
        if role == SEER:
            return (
                "Night action — Seer:\n"
                "You may either inspect ONE other player's card, OR inspect TWO center cards, OR pass.\n"
                "Respond with exactly one of (X = a player index 0..4 other than you;\n"
                "center pairs are {0,1}, {0,2}, {1,2}):\n"
                "<action>peek_player X</action>\n"
                "<action>peek_center 0 1</action>\n"
                "<action>peek_center 0 2</action>\n"
                "<action>peek_center 1 2</action>\n"
                "<action>pass</action>"
            )
        if role == ROBBER:
            return (
                "Night action — Robber:\n"
                "You may swap your card with one other player's and then look at your new card, OR pass.\n"
                "Respond with exactly one of:\n"
                "<action>swap X</action>   (X = a player index 0..4, not you)\n"
                "<action>pass</action>"
            )
        if role == TROUBLEMAKER:
            return (
                "Night action — Troublemaker:\n"
                "You may swap two OTHER players' cards without looking, OR pass.\n"
                "Respond with exactly one of:\n"
                "<action>swap X Y</action>   (X, Y distinct player indices 0..4, neither is you)\n"
                "<action>pass</action>"
            )
        return "Night action — no action available. Respond with <action>none</action>."

    def _day_prompt(self, state: ONWState, player: int) -> str:
        slots_left = sum(1 for p in state.day_queue if p == player)
        return (
            f"Day phase. You have {slots_left} speaking slot(s) remaining.\n"
            "Send one message to all players: claim a role, share information, accuse, etc.\n"
            "Be strategic — werewolves should bluff a village role; village should expose werewolves.\n"
            "Respond with:\n"
            "<message>your one-paragraph message</message>"
        )

    def _vote_prompt(self, state: ONWState, player: int) -> str:
        return (
            "Vote phase. Choose ONE other player to eliminate.\n"
            "Respond with exactly:\n"
            "<vote>X</vote>   (X = a player index 0..4 other than you)"
        )

    # --------------------------------------------------------------- parsing
    _ACT_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
    _MSG_RE = re.compile(r"<message>(.*?)</message>", re.IGNORECASE | re.DOTALL)
    _VOTE_RE = re.compile(r"<vote>\s*(\d+)\s*</vote>", re.IGNORECASE)

    def parse_action(self, state: ONWState, player: int, text: str) -> Action:
        if state.phase == PHASE_NIGHT and state.night_queue:
            return self._parse_night(state, player, text)
        if state.phase == PHASE_DAY:
            m = self._MSG_RE.search(text)
            if not m:
                raise ParseError("expected <message>...</message>")
            return {"type": "speak", "message": m.group(1).strip()}
        if state.phase == PHASE_VOTE:
            m = self._VOTE_RE.search(text)
            if not m:
                raise ParseError("expected <vote>X</vote>")
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
        if role == WEREWOLF:
            others = [i for i, r in enumerate(state.initial_roles)
                      if r == WEREWOLF and i != player]
            if others:
                if body != "acknowledge":
                    raise ParseError("werewolf must respond <action>acknowledge</action>")
                return {"type": "wolf_acknowledge"}
            # Lone wolf
            if body == "skip":
                return {"type": "wolf_no_peek"}
            mc = re.match(r"peek_center\s+([012])$", body)
            if mc:
                return {"type": "wolf_peek_center", "center": int(mc.group(1))}
            raise ParseError("expected peek_center N or skip")
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
                pair = tuple(sorted((a, b)))
                return {"type": "seer_peek_center", "pair": list(pair)}
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
        return {"type": "no_action"}

    # --------------------------------------------------------------- step
    def step(self, state: ONWState, action: Action) -> ONWState:
        t = action.get("type")

        # Allow the runner to push us through empty phases.
        if t == "advance_phase":
            return self._advance_phase(state)

        if state.phase == PHASE_NIGHT and state.night_queue:
            seat, role = state.night_queue[0]
            new = state
            if t == "wolf_acknowledge":
                others = [i for i, r in enumerate(state.initial_roles)
                          if r == WEREWOLF]
                new.night_results[seat] = (
                    f"You confirmed werewolves are players {others}."
                )
            elif t == "wolf_no_peek":
                new.night_results[seat] = "You chose not to peek at any center card."
            elif t == "wolf_peek_center":
                c = action["center"]
                new.night_results[seat] = f"Center card {c} is {state.center_current[c]}."
            elif t == "seer_pass":
                new.night_results[seat] = "You passed."
            elif t == "seer_peek_player":
                target = action["target"]
                new.night_results[seat] = (
                    f"Player {target}'s card is {state.current_roles[target]}."
                )
            elif t == "seer_peek_center":
                a, b = action["pair"]
                new.night_results[seat] = (
                    f"Center {a} = {state.center_current[a]}; "
                    f"Center {b} = {state.center_current[b]}."
                )
            elif t == "robber_pass":
                new.night_results[seat] = "You passed."
            elif t == "robber_swap":
                target = action["target"]
                # Swap robber's current card with target's current card.
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
                new.night_results[seat] = (
                    f"You swapped players {a} and {b} (no peek)."
                )
            else:
                new.night_results[seat] = "(unrecognised night action; noop)"
            new.night_queue = new.night_queue[1:]
            if not new.night_queue:
                new.phase = PHASE_DAY
            return new

        if state.phase == PHASE_DAY:
            if t == "speak":
                new = state
                speaker = new.day_queue[0]
                new.day_log.append((speaker, action.get("message", "")))
                new.day_queue = new.day_queue[1:]
                if not new.day_queue:
                    new.phase = PHASE_VOTE
                return new

        if state.phase == PHASE_VOTE:
            if t == "vote":
                new = state
                voter = new.vote_queue[0]
                new.votes[voter] = int(action["target"])
                new.vote_queue = new.vote_queue[1:]
                if not new.vote_queue:
                    new = self._resolve(new)
                return new

        return state

    def _advance_phase(self, state: ONWState) -> ONWState:
        """Empty-queue safety net."""
        if state.phase == PHASE_NIGHT and not state.night_queue:
            state.phase = PHASE_DAY
        elif state.phase == PHASE_DAY and not state.day_queue:
            state.phase = PHASE_VOTE
        elif state.phase == PHASE_VOTE and not state.vote_queue:
            state = self._resolve(state)
        return state

    def _resolve(self, state: ONWState) -> ONWState:
        # Tally votes -> find max -> eliminate all tied
        tally: Dict[int, int] = {i: 0 for i in range(self.n_players)}
        for v in state.votes.values():
            tally[v] = tally.get(v, 0) + 1
        max_votes = max(tally.values()) if tally else 0
        eliminated = [i for i, c in tally.items() if c == max_votes and c > 0]
        state.eliminated = eliminated

        # Determine winner based on current roles.
        # Canonical ONW resolution:
        #   - If at least one Werewolf is in play and one dies: village wins.
        #   - If at least one Werewolf is in play and no Werewolf dies: wolves win.
        #   - If NO Werewolves are in play (both ended up in the centre):
        #       * no one dies -> village wins (correctly identified there
        #                                       were no wolves to hunt);
        #       * someone dies -> "nobody" wins (unjust execution; everyone
        #                                       gets 0 reward).
        wolves_in_play = [i for i, r in enumerate(state.current_roles) if r == WEREWOLF]
        if not wolves_in_play:
            state.winner_team = "village" if not eliminated else "nobody"
        elif any(state.current_roles[i] == WEREWOLF for i in eliminated):
            state.winner_team = "village"
        else:
            state.winner_team = "werewolves"
        state.phase = PHASE_TERMINAL
        return state

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: ONWState) -> bool:
        return state.phase == PHASE_TERMINAL

    def rewards(self, state: ONWState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        if state.winner_team == "nobody":
            return [0.0] * self.n_players
        team_per_player = [
            "werewolves" if state.current_roles[i] == WEREWOLF else "village"
            for i in range(self.n_players)
        ]
        return [1.0 if team_per_player[i] == state.winner_team else 0.0
                for i in range(self.n_players)]
