"""Monopoly Lite — a compact 20-square "plain game" for the play harness.

A lean Monopoly variant on a 20-square board (4 colour groups of 2, 4
railroads, GO, 4 tax squares, 2 free-parking / jail-visit squares).  This is
the *plain-game* rebuild: there is **no** negotiation phase, **no** alliances,
and **no** ``roll`` action.  Each turn the engine auto-rolls the dice inside
``_begin_turn``, resolves the landing, optionally opens a buy decision, and
then offers a single end-of-turn trade dialogue before advancing.

Board
-----
sq  0: GO (collect $200 salary on pass/land)
sq  1: PURPLE property  purple1  $60  rent $10  full-group $20
sq  2: RAILROAD         rr1      $150 rent $25  full-set   $50
sq  3: PURPLE property  purple2  $60  rent $10  full-group $20
sq  4: Tax $50
sq  5: Tax $100
sq  6: LIGHT_BLUE       lblue1   $100 rent $15  full-group $30
sq  7: RAILROAD         rr2      $150 rent $25  full-set   $50
sq  8: LIGHT_BLUE       lblue2   $100 rent $15  full-group $30
sq  9: Jail Visit (nothing)
sq 10: Free Parking (nothing)
sq 11: ORANGE property  orange1  $140 rent $20  full-group $40
sq 12: RAILROAD         rr3      $150 rent $25  full-set   $50
sq 13: ORANGE property  orange2  $140 rent $20  full-group $40
sq 14: Tax $75
sq 15: Free Parking (nothing)
sq 16: RED property     red1     $200 rent $25  full-group $50
sq 17: RAILROAD         rr4      $150 rent $25  full-set   $50
sq 18: RED property     red2     $200 rent $25  full-group $50
sq 19: Luxury Tax $100

Turn machine (binding contract — consumed by the parsing / metrics layers)
--------------------------------------------------------------------------
``_begin_turn(state)`` (also run once by ``initial_state`` for seat 0):
  * roll the seeded dice, move, award GO salary on a pass/land wrap;
  * resolve the landed square:
      - tax  -> ``_pay`` to the bank (bankruptcy as usual);
      - rent -> ``_pay`` to the owner (bankruptcy as usual);
      - own / free / jail squares -> no-op;
      - unowned buyable & cash >= price -> ``phase=PH_BUY`` (decision owner is
        the roller), ``pending_buy_square`` set, and the machine STOPS here;
      - unowned buyable & cash < price  -> log ``"{name} cannot afford {pid}
        (${price})"`` and fall through.
  * if the roller went bankrupt during resolution -> ``_advance_turn``;
  * otherwise -> ``phase=PH_TRADE_PROPOSE`` (still the roller's decision).

``step`` transitions:
  * ``buy`` / ``decline`` (PH_BUY): resolve the deed, then ``PH_TRADE_PROPOSE``
    for the SAME player (never advance here).
  * ``no_trade`` (PH_TRADE_PROPOSE): ``_advance_turn``.
  * ``propose_trade`` (PH_TRADE_PROPOSE, once per turn): store ``pending_trade``
    (incl. ``message``), log the offer line (+ a spoken-message line if any),
    ``phase=PH_TRADE_RESPOND`` (decision owner is the recipient).
  * ``accept_trade`` (PH_TRADE_RESPOND): if the recipient cannot cover
    ``want_cash`` -> log ``"trade failed: insufficient funds"`` and treat as a
    rejection; otherwise execute the exact transfers (no clamping) and log a
    ``"Trade completed: ..."`` line.  Then clear ``pending_trade`` and
    ``_advance_turn``.
  * ``reject_trade`` (PH_TRADE_RESPOND): log ``"{recipient} rejected the
    trade"``, clear ``pending_trade`` and ``_advance_turn``.
  Both responses may carry an optional ``message`` (logged as a spoken line).

``_advance_turn``: pick the next solvent seat, ``turn += 1``,
``traded_this_turn = False``; if ``turn >= turn_cap`` declare the turn-cap
winner and go terminal, otherwise ``_begin_turn`` for the new player.

Turn-cap winner tie-break
-------------------------
The turn-cap winner is the solvent seat with the greatest ``_net_worth``
(cash + list prices of owned properties).  Ties are broken by **higher cash**
first, then by **lower seat index**.  Concretely the winner maximises the key
``(net_worth, cash, -seat_index)``.

Events
------
Every event line is appended to BOTH ``state.events`` (the permanent log,
tailed by ``god_view``) and ``state.new_events`` (the per-batch delta).
``observations()`` is the sole drainer of ``new_events``: it returns one
all-seat ``{"type": "event", "text": line}`` Obs per queued line and then
clears the queue.  Because the runner calls ``observations`` exactly once after
every ``step``, the queue is effectively cleared each step; the first
``_begin_turn`` run by ``initial_state`` leaves its events queued until the
first step's ``observations`` call drains them.  Trade proposal / response
steps additionally emit an all-seat ``{"type": "trade_dialogue", ...}`` Obs —
offers are public.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..base import Action, Game, Obs, ParseError
from ..config import GameConfig


# --------------------------------------------------------------------------- #
# Board definition
# --------------------------------------------------------------------------- #

BOARD_SIZE = 20
STARTING_CASH = 1500
GO_SALARY = 200

# Immutable board spec (tuple of frozen dicts).
BOARD: tuple = (
    {"sq": 0,  "type": "go"},
    {"sq": 1,  "type": "property", "prop_id": "purple1",
     "group": "PURPLE",     "price": 60,  "rent": 10, "full_rent": 20},
    {"sq": 2,  "type": "railroad", "prop_id": "rr1",
     "group": "RAILROAD",   "price": 150, "rent": 25, "full_rent": 50},
    {"sq": 3,  "type": "property", "prop_id": "purple2",
     "group": "PURPLE",     "price": 60,  "rent": 10, "full_rent": 20},
    {"sq": 4,  "type": "tax",  "amount": 50},
    {"sq": 5,  "type": "tax",  "amount": 100},
    {"sq": 6,  "type": "property", "prop_id": "lblue1",
     "group": "LIGHT_BLUE", "price": 100, "rent": 15, "full_rent": 30},
    {"sq": 7,  "type": "railroad", "prop_id": "rr2",
     "group": "RAILROAD",   "price": 150, "rent": 25, "full_rent": 50},
    {"sq": 8,  "type": "property", "prop_id": "lblue2",
     "group": "LIGHT_BLUE", "price": 100, "rent": 15, "full_rent": 30},
    {"sq": 9,  "type": "jail_visit"},
    {"sq": 10, "type": "free_parking"},
    {"sq": 11, "type": "property", "prop_id": "orange1",
     "group": "ORANGE",     "price": 140, "rent": 20, "full_rent": 40},
    {"sq": 12, "type": "railroad", "prop_id": "rr3",
     "group": "RAILROAD",   "price": 150, "rent": 25, "full_rent": 50},
    {"sq": 13, "type": "property", "prop_id": "orange2",
     "group": "ORANGE",     "price": 140, "rent": 20, "full_rent": 40},
    {"sq": 14, "type": "tax",  "amount": 75},
    {"sq": 15, "type": "free_parking"},
    {"sq": 16, "type": "property", "prop_id": "red1",
     "group": "RED",        "price": 200, "rent": 25, "full_rent": 50},
    {"sq": 17, "type": "railroad", "prop_id": "rr4",
     "group": "RAILROAD",   "price": 150, "rent": 25, "full_rent": 50},
    {"sq": 18, "type": "property", "prop_id": "red2",
     "group": "RED",        "price": 200, "rent": 25, "full_rent": 50},
    {"sq": 19, "type": "tax",  "amount": 100},
)

# Colour / railroad groups: {group -> [prop_ids in group]}
GROUPS: Dict[str, List[str]] = {
    "PURPLE":     ["purple1", "purple2"],
    "LIGHT_BLUE": ["lblue1",  "lblue2"],
    "ORANGE":     ["orange1", "orange2"],
    "RED":        ["red1",    "red2"],
    "RAILROAD":   ["rr1",     "rr2",    "rr3", "rr4"],
}

# Quick lookup: prop_id -> square dict (for all buyable squares).
PROP_INFO: Dict[str, dict] = {
    sq["prop_id"]: sq
    for sq in BOARD
    if "prop_id" in sq
}

# Phases (plain game — no negotiation, no alliance sub-phase).
PH_BUY = "buy_decision"
PH_TRADE_PROPOSE = "trade_propose"
PH_TRADE_RESPOND = "trade_respond"
PH_TERMINAL = "terminal"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

@dataclass
class MLState:
    """Full game state for the plain-game MonopolyLite.

    ``player_names`` is the authoritative seat order.  All per-player dicts are
    keyed by player name (string); ``bankrupt`` is a set of names.  The integer
    index into ``player_names`` is the "seat" the runner uses.
    """

    player_names: List[str] = field(default_factory=list)
    cash: Dict[str, int] = field(default_factory=dict)
    positions: Dict[str, int] = field(default_factory=dict)
    # prop_id -> owner name, or None if unowned.
    properties: Dict[str, Optional[str]] = field(default_factory=dict)
    bankrupt: Set[str] = field(default_factory=set)
    current_player_idx: int = 0
    phase: str = PH_TRADE_PROPOSE
    pending_buy_square: Optional[int] = None
    # Pending trade: {from, to, give_props, give_cash, want_props, want_cash,
    #                 message}.
    pending_trade: Optional[Dict] = None
    traded_this_turn: bool = False
    turn: int = 0          # incremented once per full player-turn
    turn_cap: int = 200
    game_seed: int = 0     # propagated from MonopolyLite._seed for dice
    winner: Optional[str] = None
    dice_roll: Optional[int] = None
    events: List[str] = field(default_factory=list)      # permanent log
    new_events: List[str] = field(default_factory=list)  # per-batch delta


# --------------------------------------------------------------------------- #
# Game class
# --------------------------------------------------------------------------- #

class MonopolyLite(Game):
    """Plain-game Monopoly Lite.  See module docstring for the turn machine."""

    name = "monopoly_lite"
    n_players = 4  # default; overridden by ``__init__``

    def __init__(
        self,
        players: "Optional[List[str]]" = None,
        seed: int = 0,
        turn_cap: int = 200,
        config: "Optional[GameConfig]" = None,
    ) -> None:
        super().__init__(config)
        if players is None:
            players = [f"P{i}" for i in range(4)]
        self.players: List[str] = list(players)
        self.n_players: int = len(self.players)
        self._seed: int = seed
        self._turn_cap: int = turn_cap
        # Transient: the trade dict most recently proposed / resolved by
        # ``step`` so ``observations`` can attach it to the trade_dialogue Obs
        # even after ``pending_trade`` has been cleared.  Sequential, never
        # concurrent — the runner calls observations() right after each step().
        self._last_trade: Optional[Dict] = None

    # --------------------------------------------------------------- events
    def _emit(self, state: MLState, line: str) -> None:
        """Append an event line to both the permanent log and the per-batch
        delta that ``observations`` drains."""
        state.events.append(line)
        state.new_events.append(line)

    # ------------------------------------------------------------------ setup
    def initial_state(self, rng) -> MLState:  # type: ignore[override]
        props: Dict[str, Optional[str]] = {
            sq["prop_id"]: None for sq in BOARD if "prop_id" in sq
        }
        state = MLState(
            player_names=list(self.players),
            cash={p: STARTING_CASH for p in self.players},
            positions={p: 0 for p in self.players},
            properties=props,
            bankrupt=set(),
            current_player_idx=0,
            phase=PH_TRADE_PROPOSE,
            pending_buy_square=None,
            pending_trade=None,
            traded_this_turn=False,
            turn=0,
            turn_cap=self._turn_cap,
            game_seed=self._seed,
            winner=None,
            dice_roll=None,
            events=[],
            new_events=[],
        )
        # First turn is auto-rolled; its events sit in ``new_events`` until the
        # first step's ``observations`` call drains them (intended).
        self._begin_turn(state)
        return state

    # --------------------------------------------------------------- helpers
    def living_seats(self, state: MLState) -> List[int]:
        """Non-bankrupt seat indices, preserving seat order."""
        return [
            i for i, name in enumerate(state.player_names)
            if name not in state.bankrupt
        ]

    def _roll_dice(self, state: MLState) -> int:
        """Deterministic 2d6 roll seeded from (game_seed, turn, player_idx).

        Incorporating ``game_seed`` ensures different MonopolyLite seeds
        produce distinct dice trajectories.  A local ``random.Random`` avoids
        side-effecting any global RNG.
        """
        seed = state.game_seed * 97 + state.turn * 31 + state.current_player_idx * 7
        r = random.Random(seed)
        return r.randint(1, 6) + r.randint(1, 6)

    def _net_worth(self, state: MLState, name: str) -> int:
        """Cash + sum of list prices of all owned properties."""
        worth = state.cash.get(name, 0)
        for pid, owner in state.properties.items():
            if owner == name:
                worth += PROP_INFO[pid]["price"]
        return worth

    def _compute_rent(self, state: MLState, sq: dict, owner: str) -> int:
        """Rent for ``sq``.  Doubles to ``full_rent`` when ``owner`` holds
        every property in the colour / railroad group."""
        group = sq.get("group")
        if group and group in GROUPS:
            if all(state.properties.get(p) == owner for p in GROUPS[group]):
                return int(sq["full_rent"])
        return int(sq["rent"])

    def _pay(
        self,
        state: MLState,
        payer: str,
        creditor: Optional[str],
        amount: int,
    ) -> None:
        """Pay ``amount`` from ``payer`` to ``creditor`` (``None`` == bank).

        If the payer's cash falls short they go bankrupt immediately.
        """
        if state.cash.get(payer, 0) >= amount:
            state.cash[payer] = state.cash[payer] - amount
            if creditor is not None:
                state.cash[creditor] = state.cash.get(creditor, 0) + amount
            who = "bank" if creditor is None else creditor
            self._emit(state, f"{payer} paid ${amount} to {who}")
        else:
            self._bankrupt(state, payer, creditor)

    def _bankrupt(
        self,
        state: MLState,
        player: str,
        creditor: Optional[str],
    ) -> None:
        """Declare ``player`` bankrupt; transfer all cash + properties to
        ``creditor`` (``None`` == assets vanish to the bank).  Sets the winner
        and terminal phase when only one solvent player remains.
        """
        cash = state.cash.get(player, 0)
        if creditor is not None and cash > 0:
            state.cash[creditor] = state.cash.get(creditor, 0) + cash
        state.cash[player] = 0

        for pid in list(state.properties.keys()):
            if state.properties[pid] == player:
                state.properties[pid] = creditor  # None == back to bank

        state.bankrupt.add(player)
        who = "bank" if creditor is None else creditor
        self._emit(
            state, f"{player} is BANKRUPT — assets transferred to {who}"
        )

        solvent = [p for p in state.player_names if p not in state.bankrupt]
        if len(solvent) == 1:
            state.winner = solvent[0]
            state.phase = PH_TERMINAL
        elif len(solvent) == 0:
            state.phase = PH_TERMINAL  # edge case

    # -------------------------------------------------------------- turn flow
    def _begin_turn(self, state: MLState) -> None:
        """Roll for the current player, move, resolve the landing, and set the
        next decision phase (see module docstring for the binding contract)."""
        name = state.player_names[state.current_player_idx]
        roll = self._roll_dice(state)
        state.dice_roll = roll
        old_pos = state.positions[name]
        new_pos = (old_pos + roll) % BOARD_SIZE

        if old_pos + roll >= BOARD_SIZE:
            state.cash[name] = state.cash.get(name, 0) + GO_SALARY
            label = "landed on" if new_pos == 0 else "passed"
            self._emit(state, f"{name} {label} GO, collected ${GO_SALARY}")

        state.positions[name] = new_pos
        sq = BOARD[new_pos]
        sq_type = sq["type"]
        self._emit(state, f"{name} rolled {roll} → sq {new_pos} ({sq_type})")

        if sq_type == "go":
            pass  # salary already applied above
        elif sq_type == "tax":
            self._pay(state, name, None, sq["amount"])
        elif sq_type in ("jail_visit", "free_parking"):
            pass
        elif sq_type in ("property", "railroad"):
            pid = sq["prop_id"]
            owner = state.properties.get(pid)
            if owner is None:
                price = sq["price"]
                if state.cash.get(name, 0) >= price:
                    state.pending_buy_square = new_pos
                    state.phase = PH_BUY
                    return  # decision owner = the roller
                self._emit(state, f"{name} cannot afford {pid} (${price})")
                # fall through to the end-of-turn trade dialogue
            elif owner == name:
                pass  # own it — free landing
            else:
                rent = self._compute_rent(state, sq, owner)
                self._pay(state, name, owner, rent)

        # Resolution complete.
        if state.phase == PH_TERMINAL:
            return
        if name in state.bankrupt:
            self._advance_turn(state)
            return
        state.phase = PH_TRADE_PROPOSE

    def _advance_turn(self, state: MLState) -> None:
        """Advance to the next solvent seat and begin their turn, or declare
        the turn-cap winner.

        Turn-cap tie-break: the winner maximises
        ``(net_worth, cash, -seat_index)`` — highest net worth, then highest
        cash, then lowest seat index.
        """
        n = len(state.player_names)
        idx = state.current_player_idx
        found = False
        for _ in range(n):
            idx = (idx + 1) % n
            if state.player_names[idx] not in state.bankrupt:
                found = True
                break
        if not found:
            state.phase = PH_TERMINAL  # all bankrupt (shouldn't happen)
            return

        state.current_player_idx = idx
        state.turn += 1
        state.traded_this_turn = False

        if state.turn >= state.turn_cap:
            solvent_seats = [
                i for i, nm in enumerate(state.player_names)
                if nm not in state.bankrupt
            ]
            if solvent_seats:
                best = max(
                    solvent_seats,
                    key=lambda i: (
                        self._net_worth(state, state.player_names[i]),
                        state.cash.get(state.player_names[i], 0),
                        -i,
                    ),
                )
                state.winner = state.player_names[best]
            state.phase = PH_TERMINAL
            return

        self._begin_turn(state)

    # --------------------------------------------------------- active_player
    def active_player(self, state: MLState) -> int:
        if self.is_terminal(state):
            return -1
        if state.phase == PH_TRADE_RESPOND and state.pending_trade is not None:
            recipient = state.pending_trade["to"]
            try:
                return state.player_names.index(recipient)
            except ValueError:
                return -1
        if state.phase in (PH_BUY, PH_TRADE_PROPOSE):
            return state.current_player_idx
        return -1

    # --------------------------------------------------------- legal_actions
    def legal_actions(self, state: MLState, player: int) -> List[Action]:
        if self.is_terminal(state):
            return []
        if self.active_player(state) != player:
            return []
        if state.phase == PH_BUY:
            # Unaffordability is filtered upstream (the roller only enters
            # PH_BUY when cash >= price), so both options are always legal.
            return [{"type": "decline"}, {"type": "buy"}]
        if state.phase == PH_TRADE_PROPOSE:
            # Task 4 layers proposal actions on top; the plain machine offers
            # the pass-through so a random policy always terminates a turn.
            return [{"type": "no_trade"}]
        if state.phase == PH_TRADE_RESPOND:
            return [{"type": "accept_trade"}, {"type": "reject_trade"}]
        return []

    # ------------------------------------------------------------------- step
    def step(self, state: MLState, action: Action) -> MLState:  # type: ignore[override]
        t = action.get("type", "")
        phase = state.phase

        if phase == PH_BUY:
            assert t in ("buy", "decline"), (
                f"unexpected action {t!r} in {phase}"
            )
            name = state.player_names[state.current_player_idx]
            sq = BOARD[state.pending_buy_square]  # type: ignore[index]
            pid = sq["prop_id"]
            if t == "buy":
                price = sq["price"]
                # The roller only reached PH_BUY when affordable; guard anyway.
                if state.cash.get(name, 0) >= price:
                    state.cash[name] -= price
                    state.properties[pid] = name
                    self._emit(state, f"{name} bought {pid} for ${price}")
                else:
                    self._emit(
                        state, f"{name} cannot afford {pid} (${price})"
                    )
            else:  # decline
                self._emit(state, f"{name} declined to buy {pid}")
            state.pending_buy_square = None
            state.phase = PH_TRADE_PROPOSE  # same player, NOT advance
            return state

        if phase == PH_TRADE_PROPOSE:
            assert t in ("no_trade", "propose_trade"), (
                f"unexpected action {t!r} in {phase}"
            )
            if t == "no_trade":
                self._advance_turn(state)
                return state
            # propose_trade — at most one per turn.
            assert not state.traded_this_turn, "already proposed a trade this turn"
            proposer = state.player_names[state.current_player_idx]
            recipient = action.get("to", "")
            give_props = list(action.get("give_props", []))
            give_cash = int(action.get("give_cash", 0))
            want_props = list(action.get("want_props", []))
            want_cash = int(action.get("want_cash", 0))
            message = str(action.get("message", "") or "")
            trade = {
                "from": proposer,
                "to": recipient,
                "give_props": give_props,
                "give_cash": give_cash,
                "want_props": want_props,
                "want_cash": want_cash,
                "message": message,
            }
            state.pending_trade = trade
            self._last_trade = dict(trade)
            self._emit(
                state,
                f"{proposer} proposed trade to {recipient}: "
                f"gives {give_props}+${give_cash} for {want_props}+${want_cash}",
            )
            if message:
                self._emit(state, f'{proposer} says: "{message}"')
            state.traded_this_turn = True
            state.phase = PH_TRADE_RESPOND  # recipient decides
            return state

        if phase == PH_TRADE_RESPOND:
            assert t in ("accept_trade", "reject_trade"), (
                f"unexpected action {t!r} in {phase}"
            )
            trade = state.pending_trade
            assert trade is not None, "PH_TRADE_RESPOND with no pending_trade"
            proposer = trade["from"]
            recipient = trade["to"]
            message = str(action.get("message", "") or "")
            self._last_trade = dict(trade)

            if t == "reject_trade":
                self._emit(state, f"{recipient} rejected the trade")
                if message:
                    self._emit(state, f'{recipient} says: "{message}"')
                state.pending_trade = None
                self._advance_turn(state)
                return state

            # accept_trade
            give_props = list(trade.get("give_props", []))
            give_cash = int(trade.get("give_cash", 0))
            want_props = list(trade.get("want_props", []))
            want_cash = int(trade.get("want_cash", 0))

            if state.cash.get(recipient, 0) < want_cash:
                # Cannot cover the requested cash — treat as a rejection.
                self._emit(state, "trade failed: insufficient funds")
                if message:
                    self._emit(state, f'{recipient} says: "{message}"')
                state.pending_trade = None
                self._advance_turn(state)
                return state

            # Execute atomically — exact transfers, no clamping.
            for pid in give_props:
                state.properties[pid] = recipient
            state.cash[proposer] = state.cash.get(proposer, 0) - give_cash
            state.cash[recipient] = state.cash.get(recipient, 0) + give_cash
            for pid in want_props:
                state.properties[pid] = proposer
            state.cash[recipient] = state.cash.get(recipient, 0) - want_cash
            state.cash[proposer] = state.cash.get(proposer, 0) + want_cash

            self._emit(
                state,
                f"Trade completed: {proposer} gave {give_props}+${give_cash}, "
                f"{recipient} gave {want_props}+${want_cash}",
            )
            if message:
                self._emit(state, f'{recipient} says: "{message}"')
            state.pending_trade = None
            self._advance_turn(state)
            return state

        raise AssertionError(f"step called in non-decision phase {phase!r}")

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: MLState, player: int) -> str:  # type: ignore[override]
        """Minimal per-seat view.  Task 4 replaces this with the full grammar
        prompt; RandomPlayer ignores it, so this only needs to be legible."""
        name = state.player_names[player]
        owned = sorted(
            pid for pid, owner in state.properties.items() if owner == name
        )
        other_wealth = {
            state.player_names[i]: self._net_worth(state, state.player_names[i])
            for i in range(self.n_players)
            if state.player_names[i] not in state.bankrupt and i != player
        }
        recent = "\n".join(f"  - {e}" for e in state.events[-6:]) or "  (none)"

        lines = [
            "=== Monopoly Lite ===",
            f"You are {name} (seat {player}). Turn {state.turn}/{state.turn_cap}. "
            f"Phase: {state.phase}.",
            f"Your cash: ${state.cash.get(name, 0)}  "
            f"Net worth: ${self._net_worth(state, name)}",
            f"Your position: sq {state.positions.get(name, 0)}",
            f"Your properties: {owned or '(none)'}",
            f"Bankrupt players: {sorted(state.bankrupt) or '(none)'}",
            f"Other net worths: {other_wealth}",
            f"Recent events:\n{recent}",
        ]

        if state.phase == PH_BUY and state.pending_buy_square is not None:
            sq = BOARD[state.pending_buy_square]
            lines.append(
                f"You landed on {sq.get('prop_id')} "
                f"(group {sq.get('group', 'n/a')}, price ${sq['price']}, "
                f"rent ${sq['rent']}).  Buy or decline?"
            )
        elif state.phase == PH_TRADE_PROPOSE:
            lines.append(
                "End of turn: you may propose a trade or pass (no_trade)."
            )
        elif state.phase == PH_TRADE_RESPOND and state.pending_trade:
            tr = state.pending_trade
            lines.append(
                f"Trade offer from {tr['from']}: they give "
                f"props={tr['give_props']} cash=${tr['give_cash']}; they want "
                f"props={tr['want_props']} cash=${tr['want_cash']}. "
                "Accept or reject?"
            )
        return "\n".join(lines)

    def parse_action(self, state: MLState, player: int, text: str) -> Action:  # type: ignore[override]
        # Task 4 implements the text-parsing grammar for Monopoly.  Until then
        # the engine is driven by pre-parsed dict actions (RandomPlayer / bots).
        raise ParseError(
            "Monopoly text parsing is implemented in Task 4; "
            "drive the engine with pre-parsed dict actions for now"
        )

    # ----------------------------------------------------------- terminal
    def is_terminal(self, state: MLState) -> bool:
        return state.winner is not None or state.phase == PH_TERMINAL

    def rewards(self, state: MLState) -> List[float]:  # type: ignore[override]
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        winner_name = state.winner
        return [
            1.0 if state.player_names[i] == winner_name else 0.0
            for i in range(self.n_players)
        ]

    # ---------------------------------------------------------- watcher hooks
    def god_view(self, state: MLState) -> dict:
        return {
            "cash": dict(state.cash),
            "positions": dict(state.positions),
            "properties": dict(state.properties),
            "bankrupt": sorted(state.bankrupt),
            "winner": state.winner,
            "events_tail": state.events[-20:],
        }

    def snapshot(self, state: MLState) -> dict:
        return {
            "public": {
                "turn": state.turn,
                "phase": state.phase,
                "positions": dict(state.positions),
                "properties": dict(state.properties),
                "bankrupt": sorted(state.bankrupt),
                "winner": state.winner,
            },
            "hidden": {
                "cash": dict(state.cash),
            },
        }

    def render_board(self, state: MLState, *, reveal: str = "god") -> str:
        if not isinstance(state, MLState):
            return ""
        lines = [
            f"=== Monopoly Lite board (turn {state.turn}/{state.turn_cap}, "
            f"phase {state.phase}) ==="
        ]
        for sq in BOARD:
            pos = sq["sq"]
            here = [name for name, p in state.positions.items() if p == pos]
            sq_info = f"sq {pos:2d}: {sq['type']:<14}"
            if "prop_id" in sq:
                pid = sq["prop_id"]
                owner = state.properties.get(pid)
                sq_info += f" {pid:<10} owner={owner or '(bank)':<12}"
            if here:
                sq_info += f"  [{', '.join(here)}]"
            lines.append(f"  {sq_info}")
        if reveal == "god":
            lines.append(
                "Cash: " + "  ".join(
                    f"{p}=${state.cash.get(p, 0)}" for p in state.player_names
                )
            )
            lines.append(f"Winner: {state.winner or '(none)'}")
        return "\n".join(lines)

    # ---------------------------------------------------------- observations
    def observations(
        self,
        prev_state: MLState,
        new_state: MLState,
        action: Action,
        actor: int,
    ) -> List[Obs]:
        """Drain ``new_events`` into one all-seat event Obs per line, and, on a
        trade proposal / response step, append the public trade_dialogue Obs."""
        audience = list(range(self.n_players))
        obs: List[Obs] = []
        for line in new_state.new_events:
            obs.append(Obs(
                audience=list(audience),
                payload={"type": "event", "text": line},
            ))
        new_state.new_events = []  # sole drainer of the per-batch delta

        t = action.get("type", "")
        if t in ("propose_trade", "accept_trade", "reject_trade"):
            obs.append(Obs(
                audience=list(audience),
                payload={
                    "type": "trade_dialogue",
                    "event": t,
                    "trade": self._last_trade,
                    "message": str(action.get("message", "") or ""),
                },
            ))
        return obs
