"""Diplomacy-Lite — 3 powers, 6 territories, simultaneous orders with support
resolution. No-Press (no in-game messaging) for v0.

Map (6 territories, hexagonally arranged):

         T0 ----- T1
        /  \\   /  \\
       /    \\ /    \\
      T5 --- T-- --- T2     (T-- placeholder; actually:)
       \\   / \\   /
        \\ /   \\ /
         T4 ----- T3

Actually adjacency (clean):
  T0: T1, T5
  T1: T0, T2
  T2: T1, T3
  T3: T2, T4
  T4: T3, T5
  T5: T4, T0

(A 6-cycle, plus we add T0-T3, T1-T4, T2-T5 to make every territory
adjacent to 3 others; the resulting graph is the complete bipartite K33's
complement — every territory has 3 neighbours.)

Updated adjacency:
  T0: T1, T5, T3
  T1: T0, T2, T4
  T2: T1, T3, T5
  T3: T2, T4, T0
  T4: T3, T5, T1
  T5: T4, T0, T2

Setup (3 powers; each power owns 2 territories at start):
  P0: T0 (unit), T3 (unit)
  P1: T1 (unit), T4 (unit)
  P2: T2 (unit), T5 (unit)

Each player has 2 units; each round all players issue orders for both
their units; orders resolve simultaneously; surviving units occupy
captured territories.

Order types:
  HOLD: stay put, supportable by adjacent friendly units.
  MOVE target: try to move into an adjacent territory.
  SUPPORT_HOLD target: lend a +1 to target's defence.
  SUPPORT_MOVE attacker target: lend a +1 to attacker's move to target,
    valid only if supporter is adjacent to target AND attacker is
    adjacent to target.

Resolution (simplified):
  - Each move/hold gets a strength = 1 + (# of valid uncut supports).
  - A SUPPORT is "cut" if the supporting unit's territory is itself
    being attacked by ANY foreign unit, UNLESS the support is for an
    attack INTO the attacker's source territory.
  - For each contested territory (any incoming move, possibly contested
    by the holder), pick the strictly-strongest incoming move; ties
    among movers cause a standoff and all bounce.
  - A move beats a hold if mover strength > defender strength.
  - Bounced movers stay on their origin; defenders that lose are
    dislodged (destroyed in v0 — no retreats).

Win condition: control >=4 of 6 territories, OR last power with units
standing. Match limited to 8 rounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..base import Action, Game, ParseError


N_TERR = 6
N_PLAYERS = 3
MAX_ROUNDS = 8
WIN_TERRITORIES = 4

# Adjacency (set semantics)
ADJ = {
    0: frozenset({1, 5, 3}),
    1: frozenset({0, 2, 4}),
    2: frozenset({1, 3, 5}),
    3: frozenset({2, 4, 0}),
    4: frozenset({3, 5, 1}),
    5: frozenset({4, 0, 2}),
}

PH_ORDERS = "orders"
PH_TERMINAL = "terminal"


@dataclass
class DipState:
    n_players: int = N_PLAYERS
    # unit_at[t] = owner_idx or -1 if empty
    unit_at: List[int] = field(default_factory=list)
    # territory_owner[t] = power_idx (supply-centre control persists even
    # if no unit is currently there; updated whenever a unit ENDS the
    # turn on a centre)
    territory_owner: List[int] = field(default_factory=list)
    # Pending orders: one entry per owned unit, keyed by territory
    pending_orders: Dict[int, dict] = field(default_factory=dict)
    # The runner cycles through players for each round; track who's done
    orders_complete: List[bool] = field(default_factory=list)
    round_no: int = 1
    phase: str = PH_ORDERS
    history: List[str] = field(default_factory=list)
    winner: Optional[int] = None


class DiplomacyLite(Game):
    name = "diplomacy_lite"
    n_players = N_PLAYERS

    # ----------------------------------------------------------- setup
    def initial_state(self, rng) -> DipState:
        unit_at = [0, 1, 2, 0, 1, 2]
        owner = list(unit_at)
        return DipState(
            n_players=self.n_players,
            unit_at=list(unit_at),
            territory_owner=list(owner),
            pending_orders={},
            orders_complete=[False] * self.n_players,
            round_no=1,
            phase=PH_ORDERS,
        )

    # ----------------------------------------------------- active_player
    def active_player(self, state: DipState) -> int:
        if state.phase == PH_TERMINAL:
            return -1
        if state.phase == PH_ORDERS:
            for i in range(self.n_players):
                if not state.orders_complete[i] and self._has_units(state, i):
                    return i
            # All powers done -> runner will request advance_phase
            return -1
        return -1

    @staticmethod
    def _has_units(state: DipState, player: int) -> bool:
        return any(o == player for o in state.unit_at)

    @staticmethod
    def _units_of(state: DipState, player: int) -> List[int]:
        return [t for t, o in enumerate(state.unit_at) if o == player]

    # ----------------------------------------------------- legal_actions
    def legal_actions(self, state: DipState, player: int) -> List[Action]:
        """Return a small enumeration of COMPLETE order sets for this player.

        Each Action is a fully-formed `submit_orders` payload — one order
        per owned unit. We don't enumerate the full cross-product (which
        grows as 4^|units| with HOLD + 3 MOVE options each); instead we
        return a hand-curated diverse set so that downstream RandomPlayer
        choice over `legal_actions` produces interesting play. LLM
        players bypass this entirely by calling `parse_action` on
        free-text orders.
        """
        units = self._units_of(state, player)
        if not units:
            return []
        actions: List[Action] = []
        # 1. All HOLD
        actions.append({"type": "submit_orders",
                        "orders": {u: {"type": "HOLD"} for u in units}})
        # 2. For each unit, a "this unit moves to first neighbour, others hold"
        for u in units:
            neighbours = sorted(ADJ[u])
            for nb in neighbours:
                if state.unit_at[nb] == player:
                    continue  # don't bash own unit
                orders = {v: {"type": "HOLD"} for v in units}
                orders[u] = {"type": "MOVE", "target": nb}
                actions.append({"type": "submit_orders", "orders": orders})
        # 3. All move to a random first-neighbour
        orders_all_move = {}
        for u in units:
            neighbours = [n for n in sorted(ADJ[u]) if state.unit_at[n] != player]
            if neighbours:
                orders_all_move[u] = {"type": "MOVE", "target": neighbours[0]}
            else:
                orders_all_move[u] = {"type": "HOLD"}
        actions.append({"type": "submit_orders", "orders": orders_all_move})
        return actions

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: DipState, player: int) -> str:
        head = self._public_header(state, player)
        body = self._order_prompt(state, player)
        return f"{head}\n\n{body}"

    def _public_header(self, state: DipState, player: int) -> str:
        rows = [
            f"  T{t}: unit=P{state.unit_at[t]}, supply-centre owner=P{state.territory_owner[t]}, "
            f"neighbours={sorted(ADJ[t])}"
            for t in range(N_TERR)
        ]
        history = "\n".join(f"  - {h}" for h in state.history[-10:]) if state.history else "  (none)"
        my_units = self._units_of(state, player)
        return (
            "=== Diplomacy-Lite (No-Press) ===\n"
            f"You are power P{player}. Round {state.round_no}/{MAX_ROUNDS}.\n"
            f"Your units are on: {my_units}\n"
            f"Map (6 territories, all units below):\n" + "\n".join(rows) + "\n"
            f"Recent events:\n{history}"
        )

    def _order_prompt(self, state: DipState, player: int) -> str:
        units = self._units_of(state, player)
        return (
            f"Submit ONE order per unit you own ({units}). Orders resolve "
            f"simultaneously for all powers.\n"
            "Allowed order types:\n"
            "  HOLD                         — unit stays.\n"
            "  MOVE T                       — try to move to neighbour T.\n"
            "  SUPPORT_HOLD T               — strengthen a friendly unit's hold on T.\n"
            "  SUPPORT_MOVE T_attacker T    — strengthen attacker's move into T.\n"
            "\n"
            "Format (one line per owned unit, in any order):\n"
            "<orders>\n"
            "  unit=T0 HOLD\n"
            "  unit=T3 MOVE 2\n"
            "</orders>\n"
            "\n"
            "Adjacency constraints:\n"
            "  - MOVE T: target T must be adjacent to the moving unit.\n"
            "  - SUPPORT_HOLD T: supporter's territory must be adjacent to T,\n"
            "    AND a friendly unit must be on T issuing a HOLD or non-\n"
            "    move order (otherwise the support is wasted).\n"
            "  - SUPPORT_MOVE T_attacker T: supporter must be adjacent to\n"
            "    target T, AND the attacker on T_attacker must be issuing\n"
            "    a matching MOVE T.\n"
            "Issue exactly one order per unit; omit a unit at your peril\n"
            "(missing units default to HOLD)."
        )

    # ------------------------------------------------------------- parse
    _BLOCK_RE = re.compile(r"<orders>(.*?)</orders>", re.I | re.DOTALL)
    _LINE_RE = re.compile(
        r"unit\s*=\s*T?(\d+)\s+"
        r"(HOLD|MOVE|SUPPORT_HOLD|SUPPORT_MOVE)"
        r"(?:\s+T?(\d+))?(?:\s+T?(\d+))?",
        re.I,
    )

    def parse_action(self, state: DipState, player: int, text: str) -> Action:
        if state.phase != PH_ORDERS:
            raise ParseError(f"no action expected in phase {state.phase}")
        m = self._BLOCK_RE.search(text)
        if not m:
            raise ParseError("expected an <orders>...</orders> block")
        body = m.group(1)
        orders: Dict[int, dict] = {}
        units = set(self._units_of(state, player))
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            mm = self._LINE_RE.search(line)
            if not mm:
                raise ParseError(f"could not parse order line: {line!r}")
            u = int(mm.group(1))
            otype = mm.group(2).upper()
            a = int(mm.group(3)) if mm.group(3) else None
            b = int(mm.group(4)) if mm.group(4) else None
            if u not in units:
                raise ParseError(f"you don't own a unit on T{u}")
            if otype == "HOLD":
                orders[u] = {"type": "HOLD"}
            elif otype == "MOVE":
                if a is None:
                    raise ParseError(f"MOVE needs a target on T{u}")
                if a not in ADJ[u]:
                    raise ParseError(f"T{a} not adjacent to T{u}")
                orders[u] = {"type": "MOVE", "target": a}
            elif otype == "SUPPORT_HOLD":
                if a is None:
                    raise ParseError(f"SUPPORT_HOLD needs a target on T{u}")
                if a not in ADJ[u]:
                    raise ParseError(f"T{a} not adjacent to supporter T{u}")
                orders[u] = {"type": "SUPPORT_HOLD", "target": a}
            elif otype == "SUPPORT_MOVE":
                if a is None or b is None:
                    raise ParseError(f"SUPPORT_MOVE needs attacker + target on T{u}")
                if b not in ADJ[u]:
                    raise ParseError(f"target T{b} not adjacent to supporter T{u}")
                if a not in ADJ[b]:
                    raise ParseError(f"attacker T{a} not adjacent to target T{b}")
                orders[u] = {"type": "SUPPORT_MOVE", "attacker": a, "target": b}
            else:
                raise ParseError(f"unknown order type {otype}")
        # Missing units default to HOLD
        for u in units:
            orders.setdefault(u, {"type": "HOLD"})
        return {"type": "submit_orders", "orders": orders}

    # -------------------------------------------------------------- step
    def step(self, state: DipState, action: Action) -> DipState:
        t = action.get("type")
        if t == "advance_phase":
            # All orders in; resolve.
            if state.phase == PH_ORDERS and all(
                state.orders_complete[i] or not self._has_units(state, i)
                for i in range(self.n_players)
            ):
                return self._resolve_round(state)
            return state
        if state.phase == PH_ORDERS and t == "submit_orders":
            player = self.active_player(state)
            state.pending_orders.update(action["orders"])
            state.orders_complete[player] = True
            # If everyone has submitted, resolve immediately.
            if all(state.orders_complete[i] or not self._has_units(state, i)
                   for i in range(self.n_players)):
                return self._resolve_round(state)
            return state
        return state

    def _resolve_round(self, state: DipState) -> DipState:
        orders = dict(state.pending_orders)
        # Default any missing unit to HOLD (defensive)
        for t, o in enumerate(state.unit_at):
            if o >= 0 and t not in orders:
                orders[t] = {"type": "HOLD"}

        # Build a map of who is targeted by movers
        attacks_into: Dict[int, List[int]] = {i: [] for i in range(N_TERR)}
        for src, od in orders.items():
            if od["type"] == "MOVE":
                attacks_into[od["target"]].append(src)

        # Identify SUPPORT cuts: a SUPPORT is cut iff the supporter's
        # territory is being attacked by SOME unit other than the unit
        # the support is aimed at.
        supports_valid: Dict[int, bool] = {}
        for src, od in orders.items():
            if od["type"] in ("SUPPORT_HOLD", "SUPPORT_MOVE"):
                # Who's attacking the supporter?
                attackers = [a for a in attacks_into.get(src, []) if a != src]
                if od["type"] == "SUPPORT_MOVE":
                    target = od["target"]
                    # Cut UNLESS the only attacker is the one moving INTO
                    # the support's target (i.e., the supported attacker).
                    others = [a for a in attackers if a != od["attacker"]]
                    supports_valid[src] = len(others) == 0
                else:
                    supports_valid[src] = len(attackers) == 0

        # Compute strength of each MOVE order and each HOLD.
        # MOVE strength = 1 + count(valid SUPPORT_MOVE orders pointing at this move)
        # HOLD strength = 1 + count(valid SUPPORT_HOLD orders pointing at this unit)
        move_strength: Dict[Tuple[int, int], int] = {}  # (src, target) -> str
        hold_strength: Dict[int, int] = {}              # territory -> str
        for src, od in orders.items():
            if od["type"] == "MOVE":
                base = 1
                supports = [
                    sp for sp, sod in orders.items()
                    if sod["type"] == "SUPPORT_MOVE"
                    and sod["attacker"] == src
                    and sod["target"] == od["target"]
                    and supports_valid.get(sp, False)
                ]
                move_strength[(src, od["target"])] = base + len(supports)
            elif od["type"] == "HOLD":
                supports = [
                    sp for sp, sod in orders.items()
                    if sod["type"] == "SUPPORT_HOLD"
                    and sod["target"] == src
                    and supports_valid.get(sp, False)
                ]
                hold_strength[src] = 1 + len(supports)
            elif od["type"] in ("SUPPORT_HOLD", "SUPPORT_MOVE"):
                # Supporters that aren't themselves moving HOLD their territory by default
                hold_strength[src] = 1

        # Decide each contested territory.
        new_unit_at = list(state.unit_at)
        new_units_after: Dict[int, int] = {}     # who moves where
        bounced: set = set()
        for target in range(N_TERR):
            incoming = [(src, st) for (src, t2), st in move_strength.items() if t2 == target]
            if not incoming:
                continue
            # Compare incoming strengths
            max_st = max(st for _, st in incoming)
            top = [src for src, st in incoming if st == max_st]
            defender_st = hold_strength.get(target, 0 if state.unit_at[target] < 0 else 1)
            # Possible defender is the unit currently on target (if any)
            # Defender's hold supports already included; if the current
            # unit itself is moving away, defender_st = 0 for the slot.
            if state.unit_at[target] >= 0 and orders.get(target, {}).get("type") == "MOVE":
                defender_st = 0
            if len(top) > 1 or max_st <= defender_st:
                # Standoff or insufficient strength: all incoming bounce
                bounced.update(incoming and [src for src, _ in incoming])
            else:
                winner_src = top[0]
                # The defender (if any and not moving) is dislodged
                # (destroyed in v0 since no retreats).
                new_units_after[target] = state.unit_at[winner_src]
                new_unit_at[winner_src] = -1  # source becomes empty
                # NOTE: the dislodged defender disappears (no retreat).
                if state.unit_at[target] >= 0 and orders.get(target, {}).get("type") != "MOVE":
                    pass  # destroyed

        # Apply successful moves
        for tgt, owner in new_units_after.items():
            new_unit_at[tgt] = owner

        # Units that issued MOVE but bounced stay in source UNLESS the
        # source already had another unit move in successfully (rare).
        # Our resolution above already empties src then potentially
        # repopulates if its move succeeded; bounced source slots stay
        # as the prior owner. We need to put them back.
        for src, od in orders.items():
            if od["type"] == "MOVE":
                tgt = od["target"]
                # If the move at (src, tgt) didn't win, restore src.
                if new_units_after.get(tgt) != state.unit_at[src]:
                    new_unit_at[src] = state.unit_at[src]

        # Update supply-centre ownership: any territory that ENDS with a
        # unit on it is owned by that unit's power; territories that end
        # empty retain their previous owner.
        new_owner = list(state.territory_owner)
        for t in range(N_TERR):
            if new_unit_at[t] >= 0:
                new_owner[t] = new_unit_at[t]

        state.unit_at = new_unit_at
        state.territory_owner = new_owner
        state.pending_orders = {}
        state.orders_complete = [False] * self.n_players

        # Log brief summary
        owns = [sum(1 for o in state.territory_owner if o == p) for p in range(self.n_players)]
        state.history.append(
            f"Round {state.round_no} resolved. SCs: " + " ".join(
                f"P{p}={c}" for p, c in enumerate(owns)
            )
        )
        state.round_no += 1
        # Win check
        for p in range(self.n_players):
            if owns[p] >= WIN_TERRITORIES:
                state.winner = p
                state.phase = PH_TERMINAL
                return state
        alive_powers = [p for p in range(self.n_players) if self._has_units(state, p)]
        if len(alive_powers) == 1:
            state.winner = alive_powers[0]
            state.phase = PH_TERMINAL
            return state
        if state.round_no > MAX_ROUNDS:
            state.winner = max(range(self.n_players),
                               key=lambda p: (owns[p], -p))
            state.phase = PH_TERMINAL
            return state
        return state

    # ----------------------------------------------------------- terminal
    def is_terminal(self, state: DipState) -> bool:
        return state.phase == PH_TERMINAL

    def rewards(self, state: DipState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        return [1.0 if state.winner == p else 0.0 for p in range(self.n_players)]
