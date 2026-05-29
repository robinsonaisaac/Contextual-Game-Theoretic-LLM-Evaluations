"""Risk — full 42-territory classic conquest with dice combat, cards, and a
Full-Press negotiation / alliance layer.

This is the full-map upgrade of the original toy 6-territory ``RiskLite``. The
class name and export are kept (``RiskLite``) so the game registry and existing
tests keep importing it; the public ``name`` is ``"risk"``.

Map / data
----------
The board is the standard Hasbro 42-territory map, sourced as pure data from
``game_theory_llm.play.maps.risk_map`` (``TERRITORIES``, ``ADJ``,
``CONTINENTS``, ``SET_VALUES``, ``SET_INCREMENT``). No board data lives here.

Players
-------
Default ``n_players = 4``. The initial state randomly distributes all 42
territories round-robin over a shuffled permutation and scatters each player's
starting army pool over their territories (classic 4p start = 30 armies each,
one already placed per owned territory + the remainder scattered).

Turn structure (per player)
---------------------------
1. ``PH_NEGOTIATION``  — Full-Press talk + alliance diplomacy (say / whisper /
   pass + alliance_propose/accept/decline/break). Also a pre-game round 0.
2. ``PH_DEPLOY``       — receive ``max(3, territories // 3)`` reinforcements
   + continent bonuses + any Risk-card set traded in (escalating set values
   ``4,6,8,10,12,15`` then ``+5``; forced trade when holding 5+ cards). Place
   them territory-by-territory.
3. ``PH_ATTACK``       — attack-until-you-stop: repeatedly attack from an owned
   territory (>=2 armies) into an adjacent enemy territory, until ``end_attack``.
   Capturing >=1 territory in a turn earns one Risk card.
4. ``PH_FORTIFY``      — optionally move armies once along an owned-connected
   path from one owned territory to another, then end the turn.

Elimination captures the eliminated player's Risk cards. Standard dice combat
(attacker rolls min(armies-1,3); defender rolls min(armies,2); pair high dice;
ties to defender) recorded via ``combat`` observation records.

Alliances
---------
Kinds: ``nonaggression`` / ``mutual_defense`` / ``dmz`` / ``coalition``.
``config.enforce_alliances`` (default False): attacking an *active* ally is
LEGAL and auto-breaks the alliance, emitting a ``betrayed`` alliance_event so
betrayal is measurable. If True, ally-violating attacks are filtered from
``legal_actions`` and rejected by ``parse_action`` (the "can't betray" arm).

Termination
-----------
Win by controlling all 42 territories, OR — HARD CAP — after ``MAX_ROUNDS``
(40) rounds the player holding the most territories wins (ties broken by lowest
seat). The cap guarantees a RandomPlayer-vs-RandomPlayer match always
terminates with a valid winner well within ``max_turns=400``.

Simplification (documented, behaviourally faithful)
---------------------------------------------------
Dice RNG is seeded deterministically per attack from the state rather than
drawn from the runner RNG, so combat is reproducible. Card *types*
(infantry/cavalry/artillery) are tracked and a valid set (three-of-a-kind or
one-of-each, with wild-free rules) is auto-selected when trading; the engine
trades the first legal set it finds.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..base import GOD, Action, Game, Obs, ParseError
from ..config import GameConfig
from ..messaging import MessagingMixin, NegotiationState
from ..alliances import AllianceMixin, AllianceState, new_event
from ..maps.risk_map import (
    TERRITORIES,
    ADJ,
    CONTINENTS,
    SET_VALUES,
    SET_INCREMENT,
    set_value,
)


N_TERRITORIES = len(TERRITORIES)          # 42
DEFAULT_N_PLAYERS = 4
MAX_ROUNDS = 40                           # hard cap -> winner = most territories
FORCED_TRADE_AT = 5                       # must trade a set when holding >= 5 cards
CARD_TYPES = ("infantry", "cavalry", "artillery")

# Phase tags
PH_NEGOTIATION = "negotiation"
PH_DEPLOY = "deploy"
PH_ATTACK = "attack"
PH_FORTIFY = "fortify"
PH_TERMINAL = "terminal"

# Alliance kinds this game understands.
ALLIANCE_KINDS = ("nonaggression", "mutual_defense", "dmz", "coalition")


def _starting_armies(n_players: int) -> int:
    """Classic Risk starting army pool by player count."""
    return {2: 40, 3: 35, 4: 30, 5: 25, 6: 20}.get(n_players, 30)


@dataclass
class RLState:
    n_players: int = DEFAULT_N_PLAYERS
    owner: List[int] = field(default_factory=list)      # owner[t] in [0..n_players)
    armies: List[int] = field(default_factory=list)     # armies[t] >= 1 on owned
    current_player: int = 0
    phase: str = PH_NEGOTIATION
    armies_to_deploy: int = 0
    round_no: int = 1
    turn: int = 0                                        # global step counter
    eliminated: List[bool] = field(default_factory=list)
    cards: List[List[str]] = field(default_factory=list)  # per-player list of card types
    sets_traded: int = 0                                 # global count of sets cashed in
    captured_this_turn: bool = False                     # earns a card at end of turn
    history: List[str] = field(default_factory=list)
    winner: Optional[int] = None
    win_reason: str = ""
    last_attack_roll: Optional[dict] = None

    # Shared messaging / alliance sub-state.
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)


class RiskLite(MessagingMixin, AllianceMixin, Game):
    name = "risk"
    n_players = DEFAULT_N_PLAYERS

    def __init__(self, config: "Optional[GameConfig]" = None,
                 n_players: "Optional[int]" = None) -> None:
        super().__init__(config)
        if n_players is not None:
            self.n_players = int(n_players)

    # ----------------------------------------------------------- setup
    def initial_state(self, rng) -> RLState:
        n = self.n_players
        # Random territory distribution: shuffle all 42, deal round-robin.
        order = list(range(N_TERRITORIES))
        rng.shuffle(order)
        owner = [0] * N_TERRITORIES
        for i, t in enumerate(order):
            owner[t] = i % n
        # One army on every territory, then scatter the remaining pool.
        armies = [1] * N_TERRITORIES
        pool = _starting_armies(n)
        for p in range(n):
            owned = [t for t in range(N_TERRITORIES) if owner[t] == p]
            remaining = pool - len(owned)
            for _ in range(max(0, remaining)):
                armies[rng.choice(owned)] += 1
        state = RLState(
            n_players=n,
            owner=owner,
            armies=armies,
            current_player=0,
            phase=PH_NEGOTIATION,
            round_no=1,
            turn=0,
            eliminated=[False] * n,
            cards=[[] for _ in range(n)],
            sets_traded=0,
            captured_this_turn=False,
        )
        # Pre-game round-0 negotiation, then P0 deploys.
        self.start_negotiation(state, return_phase=PH_DEPLOY,
                               rounds=self.config.nego_rounds)
        state.armies_to_deploy = self._reinforcements_for(state, 0)
        return state

    # ---- reinforcements -------------------------------------------------
    def _territory_count(self, state: RLState, player: int) -> int:
        return sum(1 for o in state.owner if o == player)

    def _continent_bonus(self, state: RLState, player: int) -> int:
        bonus = 0
        for _name, (members, value) in CONTINENTS.items():
            if all(state.owner[t] == player for t in members):
                bonus += value
        return bonus

    def _reinforcements_for(self, state: RLState, player: int) -> int:
        terr = self._territory_count(state, player)
        return max(3, terr // 3) + self._continent_bonus(state, player)

    # ---- card-set trading ----------------------------------------------
    @staticmethod
    def _find_set(cards: List[str]) -> "Optional[List[int]]":
        """Return indices of a valid 3-card set (three-of-a-kind or one-of-each)
        in ``cards``, or None. Deterministic: prefers one-of-each, then triples.
        """
        by_type: Dict[str, List[int]] = {ct: [] for ct in CARD_TYPES}
        for i, c in enumerate(cards):
            by_type.setdefault(c, []).append(i)
        # one of each
        if all(by_type.get(ct) for ct in CARD_TYPES):
            return [by_type[ct][0] for ct in CARD_TYPES]
        # three of a kind
        for ct in CARD_TYPES:
            if len(by_type.get(ct, [])) >= 3:
                return by_type[ct][:3]
        return None

    def _trade_set(self, state: RLState, player: int) -> int:
        """Trade in the first valid set the player holds. Returns army bonus
        (0 if no set). Mutates ``state.cards`` and ``state.sets_traded``."""
        idxs = self._find_set(state.cards[player])
        if idxs is None:
            return 0
        bonus = set_value(state.sets_traded)
        state.sets_traded += 1
        for i in sorted(idxs, reverse=True):
            state.cards[player].pop(i)
        state.history.append(
            f"P{player} traded a card set for {bonus} armies "
            f"(set #{state.sets_traded})"
        )
        return bonus

    def _award_card(self, state: RLState, player: int) -> None:
        # Deterministic card type by current count so smoke runs stay stable.
        ct = CARD_TYPES[len(state.cards[player]) % len(CARD_TYPES)]
        state.cards[player].append(ct)

    # ------------------------------------------------------ living seats
    def living_seats(self, state: RLState) -> List[int]:
        return [p for p in range(state.n_players) if not state.eliminated[p]]

    # ------------------------------------------------------ active_player
    def active_player(self, state: RLState) -> int:
        if state.phase == PH_TERMINAL:
            return -1
        if state.phase == PH_NEGOTIATION:
            return self.nego_active_player(state)
        return state.current_player

    # ---------------------------------------------------- legal_actions
    def legal_actions(self, state: RLState, player: int) -> List[Action]:
        if state.phase == PH_NEGOTIATION:
            return self.nego_legal_actions(state, player)
        if state.phase == PH_DEPLOY:
            owned = [t for t in range(N_TERRITORIES) if state.owner[t] == player]
            return [{"type": "deploy", "territory": t, "count": 1} for t in owned]
        if state.phase == PH_ATTACK:
            acts: List[Action] = [{"type": "end_attack"}]
            enforce = bool(self.config.enforce_alliances)
            for src in range(N_TERRITORIES):
                if state.owner[src] != player or state.armies[src] < 2:
                    continue
                for dst in ADJ[src]:
                    if state.owner[dst] == player:
                        continue
                    if enforce and self.is_allied(state, player, state.owner[dst]):
                        continue
                    acts.append({"type": "attack", "src": src, "dst": dst})
            return acts
        if state.phase == PH_FORTIFY:
            acts: List[Action] = [{"type": "end_turn"}]
            for src in range(N_TERRITORIES):
                if state.owner[src] != player or state.armies[src] < 2:
                    continue
                reachable = self._connected_owned(state, src, player)
                for dst in reachable:
                    if dst == src:
                        continue
                    # Move up to armies-1 (leave one behind); enumerate a single
                    # representative count for legal-action listing.
                    acts.append({"type": "fortify", "src": src, "dst": dst,
                                 "count": state.armies[src] - 1})
            return acts
        return []

    def _connected_owned(self, state: RLState, src: int, player: int) -> List[int]:
        """All owned territories reachable from ``src`` through owned territories."""
        seen = {src}
        stack = [src]
        while stack:
            cur = stack.pop()
            for nxt in ADJ[cur]:
                if nxt not in seen and state.owner[nxt] == player:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(seen)

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: RLState, player: int) -> str:
        head = self._public_header(state, player)
        chat = self.render_message_log(state, player)
        chat_block = f"\n\nTranscript:\n{chat}" if chat else ""
        if state.phase == PH_NEGOTIATION:
            body = self._negotiation_prompt(state, player)
        elif state.phase == PH_DEPLOY:
            body = self._deploy_prompt(state, player)
        elif state.phase == PH_ATTACK:
            body = self._attack_prompt(state, player)
        elif state.phase == PH_FORTIFY:
            body = self._fortify_prompt(state, player)
        else:
            body = "[Game over.]"
        return f"{head}{chat_block}\n\n{body}"

    def _public_header(self, state: RLState, player: int) -> str:
        counts = {p: self._territory_count(state, p)
                  for p in range(state.n_players)}
        owned = [t for t in range(N_TERRITORIES) if state.owner[t] == player]
        owned_desc = ", ".join(
            f"{TERRITORIES[t]}(#{t},{state.armies[t]})" for t in owned
        ) or "(none)"
        active_alli = [
            f"#{al.id} {{{','.join('P'+str(m) for m in al.members)}}} {al.kind}"
            for al in state.alli.alliances.values() if al.status == "active"
        ]
        alli_desc = "; ".join(active_alli) if active_alli else "(none)"
        history = "\n".join(f"  - {h}" for h in state.history[-8:]) \
            if state.history else "  (none)"
        return (
            f"=== Risk (42-territory) ===\n"
            f"You are player P{player}. Current player: P{state.current_player}. "
            f"Round {state.round_no}/{MAX_ROUNDS}. Phase: {state.phase}.\n"
            f"Territory counts: {counts}.\n"
            f"Your territories: {owned_desc}.\n"
            f"Your Risk cards: {state.cards[player]}.\n"
            f"Active alliances: {alli_desc}.\n"
            f"Recent events:\n{history}"
        )

    def _negotiation_prompt(self, state: RLState, player: int) -> str:
        living = [p for p in self.living_seats(state) if p != player]
        return (
            "Negotiation phase. Talk, then deployment begins.\n"
            "You may send ONE of:\n"
            "  <say>public message</say>\n"
            "  <whisper to=2,3>private message</whisper>\n"
            "  <pass></pass>\n"
            "Alliance moves (kinds: nonaggression, mutual_defense, dmz, "
            "coalition):\n"
            f"  <ally propose to=2,3 kind=nonaggression>terms</ally>   "
            f"(living others: {living})\n"
            "  <ally accept ID>   <ally decline ID>   <ally break ID>reason</ally>"
        )

    def _deploy_prompt(self, state: RLState, player: int) -> str:
        owned = [t for t in range(N_TERRITORIES) if state.owner[t] == player]
        return (
            f"Deploy phase. You have {state.armies_to_deploy} reinforcement(s) "
            f"left to place, ONE at a time, on any territory you own.\n"
            f"Owned territory ids: {owned}.\n"
            "Respond with exactly:\n"
            "<deploy>T</deploy>   (T = an owned territory id)"
        )

    def _attack_prompt(self, state: RLState, player: int) -> str:
        attacks = []
        enforce = bool(self.config.enforce_alliances)
        for src in range(N_TERRITORIES):
            if state.owner[src] != player or state.armies[src] < 2:
                continue
            for dst in ADJ[src]:
                if state.owner[dst] == player:
                    continue
                if enforce and self.is_allied(state, player, state.owner[dst]):
                    continue
                attacks.append(
                    f"  from {TERRITORIES[src]}(#{src},{state.armies[src]}) "
                    f"into {TERRITORIES[dst]}(#{dst},P{state.owner[dst]},"
                    f"{state.armies[dst]})"
                )
        lines = "\n".join(attacks) or "  (no legal attacks)"
        return (
            "Attack phase (attack as many times as you like, then stop).\n"
            f"Legal attacks:\n{lines}\n"
            "Respond with EXACTLY ONE of:\n"
            "<attack src=S dst=D></attack>   (S, D = territory ids)\n"
            "<end></end>   (stop attacking, move to fortify)"
        )

    def _fortify_prompt(self, state: RLState, player: int) -> str:
        return (
            "Fortify phase. You may make ONE move of armies from one owned "
            "territory to another connected owned territory, then your turn "
            "ends.\n"
            "Respond with EXACTLY ONE of:\n"
            "<fortify src=S dst=D count=N></fortify>\n"
            "<end></end>   (end turn without fortifying)"
        )

    # -------------------------------------------------------------- parse
    _DEP_RE = re.compile(r"<deploy>\s*(\d+)\s*</deploy>", re.I)
    _ATK_RE = re.compile(r"<attack\s+src=(\d+)\s+dst=(\d+)\s*>\s*</attack>", re.I)
    _FORT_RE = re.compile(
        r"<fortify\s+src=(\d+)\s+dst=(\d+)\s+count=(\d+)\s*>\s*</fortify>", re.I)
    _END_RE = re.compile(r"<end>\s*</end>", re.I)

    def parse_action(self, state: RLState, player: int, text: str) -> Action:
        if state.phase == PH_NEGOTIATION:
            return self.nego_parse(state, player, text)

        if state.phase == PH_DEPLOY:
            m = self._DEP_RE.search(text)
            if not m:
                raise ParseError("expected <deploy>T</deploy>")
            t = int(m.group(1))
            if not (0 <= t < N_TERRITORIES) or state.owner[t] != player:
                raise ParseError(f"cannot deploy on territory {t}")
            return {"type": "deploy", "territory": t, "count": 1}

        if state.phase == PH_ATTACK:
            if self._END_RE.search(text):
                return {"type": "end_attack"}
            m = self._ATK_RE.search(text)
            if not m:
                raise ParseError("expected <attack src=S dst=D></attack> or <end></end>")
            src, dst = int(m.group(1)), int(m.group(2))
            if not (0 <= src < N_TERRITORIES) or not (0 <= dst < N_TERRITORIES):
                raise ParseError("territory id out of range")
            if state.owner[src] != player:
                raise ParseError(f"you don't own territory {src}")
            if state.armies[src] < 2:
                raise ParseError(f"territory {src} needs >=2 armies to attack")
            if dst not in ADJ[src]:
                raise ParseError(f"territory {dst} is not adjacent to {src}")
            if state.owner[dst] == player:
                raise ParseError(f"territory {dst} is yours")
            if self.config.enforce_alliances and self.is_allied(
                    state, player, state.owner[dst]):
                raise ParseError(
                    f"attacking ally P{state.owner[dst]} is forbidden "
                    f"(enforce_alliances=True)")
            return {"type": "attack", "src": src, "dst": dst}

        if state.phase == PH_FORTIFY:
            if self._END_RE.search(text):
                return {"type": "end_turn"}
            m = self._FORT_RE.search(text)
            if not m:
                raise ParseError(
                    "expected <fortify src=S dst=D count=N></fortify> or <end></end>")
            src, dst, count = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (0 <= src < N_TERRITORIES) or not (0 <= dst < N_TERRITORIES):
                raise ParseError("territory id out of range")
            if state.owner[src] != player or state.owner[dst] != player:
                raise ParseError("fortify requires two owned territories")
            if src == dst:
                raise ParseError("fortify src and dst must differ")
            if dst not in self._connected_owned(state, src, player):
                raise ParseError(f"{dst} is not connected to {src} through owned land")
            if count < 1 or count > state.armies[src] - 1:
                raise ParseError(
                    f"fortify count must be 1..{state.armies[src] - 1}")
            return {"type": "fortify", "src": src, "dst": dst, "count": count}

        raise ParseError(f"no action expected in phase {state.phase}")

    # ----------------------------------------------------------------- step
    _MSG_TYPES = {"say", "whisper", "pass_talk",
                  "alliance_propose", "alliance_accept", "alliance_decline",
                  "alliance_break"}

    def step(self, state: RLState, action: Action) -> RLState:
        t = action.get("type")
        state.turn += 1

        # Negotiation / alliance actions delegate to the shared mixin.
        if t in self._MSG_TYPES:
            return self.nego_step(state, action)

        # Safety net: runner asked us to advance through a stalled phase.
        if t == "advance_phase":
            return self._advance_phase(state)

        if state.phase == PH_DEPLOY and t == "deploy":
            tgt = int(action["territory"])
            state.armies[tgt] += 1
            state.armies_to_deploy -= 1
            if state.armies_to_deploy <= 0:
                state.phase = PH_ATTACK
            return state

        if state.phase == PH_ATTACK and t == "attack":
            return self._resolve_attack(state, int(action["src"]),
                                        int(action["dst"]))

        if state.phase == PH_ATTACK and t == "end_attack":
            state.phase = PH_FORTIFY
            return state

        if state.phase == PH_FORTIFY and t == "fortify":
            src, dst, count = (int(action["src"]), int(action["dst"]),
                               int(action["count"]))
            count = min(count, state.armies[src] - 1)
            count = max(count, 1)
            state.armies[src] -= count
            state.armies[dst] += count
            state.history.append(
                f"P{state.current_player} fortified {count} "
                f"{TERRITORIES[src]}->{TERRITORIES[dst]}")
            return self._end_turn(state)

        if state.phase == PH_FORTIFY and t == "end_turn":
            return self._end_turn(state)

        return state

    def _advance_phase(self, state: RLState) -> RLState:
        if state.phase == PH_NEGOTIATION and not state.nego.speak_queue:
            self._exit_negotiation(state)
        elif state.phase == PH_DEPLOY and state.armies_to_deploy <= 0:
            state.phase = PH_ATTACK
        elif state.phase == PH_ATTACK:
            state.phase = PH_FORTIFY
        elif state.phase == PH_FORTIFY:
            return self._end_turn(state)
        return state

    # ---- combat ---------------------------------------------------------
    def _resolve_attack(self, state: RLState, src: int, dst: int) -> RLState:
        # Honour/betray judgement BEFORE the board mutates (an active ally on
        # the receiving end is a betrayal; auto-break in non-enforce mode).
        self.judge_alliance(state, {"type": "attack", "src": src, "dst": dst},
                            state.current_player)

        seed = (src * 13 + dst * 17 + state.round_no * 23
                + state.current_player * 7 + sum(state.armies) * 3
                + state.turn * 11)
        rng = random.Random(seed)
        atk_n = min(state.armies[src] - 1, 3)
        def_n = min(state.armies[dst], 2)
        atk_rolls = sorted((rng.randint(1, 6) for _ in range(atk_n)), reverse=True)
        def_rolls = sorted((rng.randint(1, 6) for _ in range(def_n)), reverse=True)
        atk_lost = def_lost = 0
        for a, d in zip(atk_rolls, def_rolls):
            if a > d:
                def_lost += 1
            else:
                atk_lost += 1
        state.armies[src] -= atk_lost
        state.armies[dst] -= def_lost
        captured = False
        if state.armies[dst] == 0:
            captured = True
            defeated_owner = state.owner[dst]
            state.owner[dst] = state.current_player
            move = min(atk_n, state.armies[src] - 1)
            move = max(move, 1)
            state.armies[src] -= move
            state.armies[dst] += move
            state.captured_this_turn = True
            state.history.append(
                f"P{state.current_player} captured {TERRITORIES[dst]} "
                f"(moved {move})")
            # Elimination: capturer takes the eliminated player's cards.
            if not any(state.owner[i] == defeated_owner
                       for i in range(N_TERRITORIES)):
                state.eliminated[defeated_owner] = True
                taken = state.cards[defeated_owner]
                state.cards[state.current_player].extend(taken)
                state.cards[defeated_owner] = []
                state.history.append(
                    f"P{defeated_owner} eliminated; P{state.current_player} "
                    f"seized {len(taken)} card(s)")
                # Forced trades while at/over the cap after seizing cards.
                while len(state.cards[state.current_player]) >= FORCED_TRADE_AT:
                    bonus = self._trade_set(state, state.current_player)
                    if bonus <= 0:
                        break
                    state.armies_to_deploy += bonus
            # Overall winner?
            owners = {state.owner[i] for i in range(N_TERRITORIES)}
            if len(owners) == 1:
                state.winner = owners.pop()
                state.win_reason = "controls all 42 territories"
                state.phase = PH_TERMINAL
                return state
        state.last_attack_roll = {
            "src": src, "dst": dst, "atk_rolls": atk_rolls, "def_rolls": def_rolls,
            "atk_lost": atk_lost, "def_lost": def_lost, "captured": captured,
        }
        state.history.append(
            f"P{state.current_player} attacked {TERRITORIES[src]}->"
            f"{TERRITORIES[dst]}: atk={atk_rolls} def={def_rolls} "
            f"A-{atk_lost} D-{def_lost}")
        # Stay in PH_ATTACK (attack-until-you-stop).
        return state

    # ---- turn rotation --------------------------------------------------
    def _end_turn(self, state: RLState) -> RLState:
        # Award a card if at least one territory was captured this turn.
        if state.captured_this_turn:
            self._award_card(state, state.current_player)
        state.captured_this_turn = False
        return self._next_player(state)

    def _next_player(self, state: RLState) -> RLState:
        nxt = (state.current_player + 1) % state.n_players
        for _ in range(state.n_players):
            if not state.eliminated[nxt]:
                break
            nxt = (nxt + 1) % state.n_players
        if nxt <= state.current_player:
            state.round_no += 1
        state.current_player = nxt

        # Hard cap -> crown the territory leader.
        if state.round_no > MAX_ROUNDS:
            counts = {p: self._territory_count(state, p)
                      for p in range(state.n_players)
                      if not state.eliminated[p]}
            if counts:
                state.winner = max(counts, key=lambda p: (counts[p], -p))
                state.win_reason = f"most territories at round cap ({MAX_ROUNDS})"
            state.phase = PH_TERMINAL
            return state

        # New turn: forced trade if holding 5+ cards, then negotiation.
        state.armies_to_deploy = 0
        while len(state.cards[nxt]) >= FORCED_TRADE_AT:
            bonus = self._trade_set(state, nxt)
            if bonus <= 0:
                break
            state.armies_to_deploy += bonus
        # Optional set trade if the player happens to hold a set (auto-cash one
        # set per turn so card armies actually enter play in RandomPlayer runs).
        if self._find_set(state.cards[nxt]) is not None:
            state.armies_to_deploy += self._trade_set(state, nxt)
        state.armies_to_deploy += self._reinforcements_for(state, nxt)

        self.start_negotiation(state, return_phase=PH_DEPLOY,
                               rounds=self.config.nego_rounds)
        return state

    # ----------------------------------------------------------- observations
    def observations(self, prev_state, new_state, action: Action,
                     actor: int) -> List[Obs]:
        t = action.get("type")
        if t in self._MSG_TYPES:
            return self.nego_observations(prev_state, new_state, action, actor)
        if t == "attack" and new_state.last_attack_roll is not None:
            r = new_state.last_attack_roll
            payload = {
                "type": "combat", "src": r["src"], "dst": r["dst"],
                "atk_rolls": r["atk_rolls"], "def_rolls": r["def_rolls"],
                "atk_lost": r["atk_lost"], "def_lost": r["def_lost"],
                "captured": r["captured"], "actor": actor,
            }
            return [Obs(audience=list(range(new_state.n_players)), payload=payload)]
        if t in ("deploy", "fortify", "end_attack", "end_turn"):
            payload = {"type": "action", "player": actor, "action": action}
            return [Obs(audience=list(range(new_state.n_players)), payload=payload)]
        return [Obs(audience=list(range(new_state.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

    # ----------------------------------------------------------- alliances
    def alliance_legal_actions(self, state: RLState, player: int) -> List[Action]:
        """Risk-specific: accept/decline pending, break active, and propose
        each alliance kind to each other living seat."""
        alli = state.alli
        actions: List[Action] = []
        for al in alli.alliances.values():
            if al.status == "proposed" and player in al.pending:
                actions.append({"type": "alliance_accept", "alliance_id": al.id})
                actions.append({"type": "alliance_decline", "alliance_id": al.id})
            if al.status == "active" and player in al.members:
                actions.append({"type": "alliance_break", "alliance_id": al.id,
                                "reason": ""})
        for other in self.living_seats(state):
            if other == player:
                continue
            for kind in ALLIANCE_KINDS:
                actions.append({"type": "alliance_propose", "to": [other],
                                "kind": kind, "terms": {}})
        return actions

    def judge_alliance(self, state: RLState, action: Action,
                       actor: int) -> List[dict]:
        """Called immediately before an attack resolves. If the defender is an
        active ally, record a ``betrayed`` event and (non-enforce mode) break
        the alliance. Honour is implicit (not attacking allies); we do not emit
        a positive ``honored`` record per non-attack to avoid log spam."""
        if action.get("type") != "attack":
            return []
        dst = int(action["dst"])
        victim = state.owner[dst]
        events: List[dict] = []
        for al in list(state.alli.alliances.values()):
            if al.status != "active":
                continue
            if actor in al.members and victim in al.members:
                # Betrayal: actor attacks an active ally.
                al.status = "broken"
                al.broken_turn = state.turn
                al.broken_by = actor
                ev = new_event("betrayed", al, turn=state.turn, actor=actor,
                               counterparty=[victim],
                               action_ref={"type": "attack",
                                           "src": int(action["src"]), "dst": dst})
                state.alli.events.append(ev)
                events.append(ev)
                state.history.append(
                    f"P{actor} BETRAYED ally P{victim} (alliance #{al.id} broken)")
        return events

    # ----------------------------------------------------------- watcher hooks
    def god_view(self, state: RLState) -> dict:
        return {
            "owner": list(state.owner),
            "armies": list(state.armies),
            "cards": [list(c) for c in state.cards],
            "eliminated": list(state.eliminated),
        }

    def snapshot(self, state: RLState) -> dict:
        public = {
            "owner": list(state.owner),
            "armies": list(state.armies),
            "territory_counts": {p: self._territory_count(state, p)
                                 for p in range(state.n_players)},
            "current_player": state.current_player,
            "round_no": state.round_no,
            "phase": state.phase,
            "alliances": [
                {"id": al.id, "members": list(al.members), "kind": al.kind,
                 "status": al.status}
                for al in state.alli.alliances.values()
            ],
        }
        hidden = {
            "cards": [list(c) for c in state.cards],
            "sets_traded": state.sets_traded,
        }
        return {"public": public, "hidden": hidden}

    def render_board(self, state: RLState, *, reveal: str = "god") -> str:
        lines = [f"=== Risk board (round {state.round_no}/{MAX_ROUNDS}, "
                 f"phase {state.phase}) ==="]
        for cont, (members, value) in CONTINENTS.items():
            owners = {state.owner[t] for t in members}
            holder = (f"  [P{next(iter(owners))} holds all, +{value}]"
                      if len(owners) == 1 else "")
            lines.append(f"{cont}:{holder}")
            for t in sorted(members):
                lines.append(
                    f"  {TERRITORIES[t]:<22} #{t:<2} P{state.owner[t]} "
                    f"x{state.armies[t]}")
        counts = {p: self._territory_count(state, p)
                  for p in range(state.n_players)}
        lines.append(f"Territory counts: {counts}")
        active = [al for al in state.alli.alliances.values()
                  if al.status == "active"]
        if active:
            lines.append("Active alliances: " + "; ".join(
                f"#{al.id}{{{','.join('P'+str(m) for m in al.members)}}}:{al.kind}"
                for al in active))
        if state.winner is not None:
            lines.append(f"WINNER: P{state.winner} ({state.win_reason})")
        return "\n".join(lines)

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: RLState) -> bool:
        return state.phase == PH_TERMINAL

    def rewards(self, state: RLState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * state.n_players
        return [1.0 if state.winner == p else 0.0
                for p in range(state.n_players)]
