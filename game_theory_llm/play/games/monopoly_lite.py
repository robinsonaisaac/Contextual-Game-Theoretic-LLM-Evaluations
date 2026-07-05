"""Monopoly Lite — compact 20-square board for multi-agent cooperation research.

A lean Monopoly variant designed to surface trading / cooperation dynamics
in a short, bounded game. The board has 20 squares covering 4 colour groups
(2 properties each), 4 railroads, GO, 4 tax squares, and 2 free-parking
squares.

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

Turn structure (per player)
---------------------------
1. ``roll`` — current player rolls 2d6 and moves.
2. Resolve landing:
   - GO: salary already awarded; advance turn.
   - Tax: pay fine (bank); check bankruptcy; advance turn.
   - Jail Visit / Free Parking: nothing; advance turn.
   - Unowned property/railroad: phase → ``buy_decision``.
   - Own property: nothing; advance turn.
   - Other's property: pay rent (full-group if owner has monopoly); check
     bankruptcy; advance turn.
3. Optional ``buy_decision`` — buy or decline the just-landed property.
4. Turn advances to next solvent player.

Trading (cooperation surface)
------------------------------
A player may call ``propose_trade`` (any phase, not in legal_actions for the
random baseline) before rolling. This puts the game in ``trade_response``
and the counterparty accepts or rejects; the proposer then rolls.

Bankruptcy
----------
A player who cannot pay rent or tax is immediately bankrupt.  All their
cash + properties transfer to the creditor (rent) or vanish to the bank
(tax).  The last solvent player wins immediately.  At ``turn_cap`` the
richest solvent player (cash + list prices of owned properties) wins.

Alliance / pact integration (MessagingMixin + AllianceMixin)
------------------------------------------------------------
``state.nego`` (NegotiationState) and ``state.alli`` (AllianceState) are
present for full mixin compatibility.  Alliance proposals (nonaggression)
are tracked via AllianceMixin; game effects are cheap-talk only.
``judge_alliance`` is a no-op (honour/betray is observable from pact ledger
via the metrics layer without mechanical enforcement).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..base import Action, Game, Obs, ParseError
from ..config import GameConfig
from ..messaging import MessagingMixin, NegotiationState
from ..alliances import AllianceMixin, AllianceState


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

# Phases
PH_ROLL = "roll"
PH_BUY = "buy_decision"
PH_TRADE = "trade_response"
PH_TERMINAL = "terminal"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

@dataclass
class MLState:
    """Full game state for Monopoly Lite.

    player_names is the authoritative seat order.  All per-player dicts
    are keyed by player name (string).  ``bankrupt`` is a set of names.
    The integer index into ``player_names`` is the "seat" used by the
    runner/mixin interfaces.
    """

    player_names: List[str] = field(default_factory=list)
    cash: Dict[str, int] = field(default_factory=dict)
    positions: Dict[str, int] = field(default_factory=dict)
    # prop_id -> owner name, or None if unowned.
    properties: Dict[str, Optional[str]] = field(default_factory=dict)
    bankrupt: Set[str] = field(default_factory=set)
    current_player_idx: int = 0
    phase: str = PH_ROLL
    # Pending trade: {from, to, give_props, give_cash, want_props, want_cash}
    pending_trade: Optional[Dict] = None
    pending_buy_square: Optional[int] = None
    turn: int = 0          # incremented once per full player-turn
    turn_cap: int = 200
    game_seed: int = 0     # propagated from MonopolyLite._seed for dice
    winner: Optional[str] = None
    dice_roll: Optional[int] = None
    history: List[str] = field(default_factory=list)
    # Mixin sub-states.
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)


# --------------------------------------------------------------------------- #
# Game class
# --------------------------------------------------------------------------- #

class MonopolyLite(MessagingMixin, AllianceMixin, Game):
    """Monopoly Lite multi-agent game.  See module docstring."""

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

    # ------------------------------------------------------------------ setup
    def initial_state(self, rng) -> MLState:  # type: ignore[override]
        props: Dict[str, Optional[str]] = {
            sq["prop_id"]: None for sq in BOARD if "prop_id" in sq
        }
        return MLState(
            player_names=list(self.players),
            cash={p: STARTING_CASH for p in self.players},
            positions={p: 0 for p in self.players},
            properties=props,
            bankrupt=set(),
            current_player_idx=0,
            phase=PH_ROLL,
            pending_trade=None,
            pending_buy_square=None,
            turn=0,
            turn_cap=self._turn_cap,
            game_seed=self._seed,
            winner=None,
            dice_roll=None,
            history=[],
            nego=NegotiationState(),
            alli=AllianceState(),
        )

    # --------------------------------------------------------------- mixins
    def living_seats(self, state: MLState) -> List[int]:
        """Non-bankrupt seat indices, preserving seat order."""
        return [
            i for i, name in enumerate(state.player_names)
            if name not in state.bankrupt
        ]

    # --------------------------------------------------------- active_player
    def active_player(self, state: MLState) -> int:
        if self.is_terminal(state):
            return -1
        if state.phase == PH_TRADE and state.pending_trade is not None:
            recipient = state.pending_trade["to"]
            try:
                return state.player_names.index(recipient)
            except ValueError:
                return -1
        if state.phase in (PH_ROLL, PH_BUY):
            name = state.player_names[state.current_player_idx]
            if name in state.bankrupt:
                return -1  # defensive; should not happen
            return state.current_player_idx
        return -1

    # --------------------------------------------------------- legal_actions
    def legal_actions(self, state: MLState, player: int) -> List[Action]:
        if self.is_terminal(state):
            return []
        active = self.active_player(state)
        if active != player:
            return []

        if state.phase == PH_ROLL:
            return [{"type": "roll"}]

        if state.phase == PH_BUY:
            player_name = state.player_names[player]
            sq = BOARD[state.pending_buy_square]  # type: ignore[index]
            actions: List[Action] = [{"type": "decline"}]
            if state.cash.get(player_name, 0) >= sq["price"]:
                actions.append({"type": "buy"})
            return actions

        if state.phase == PH_TRADE:
            return [{"type": "accept_trade"}, {"type": "reject_trade"}]

        return []

    # --------------------------------------------------------------- helpers
    def _roll_dice(self, state: MLState) -> int:
        """Deterministic 2d6 roll, seeded from (game_seed, turn, player_idx).

        Incorporating ``game_seed`` ensures different MonopolyLite seeds
        produce distinct dice trajectories (same state structure, different
        rolls). Using a local ``random.Random`` avoids side-effecting any
        global RNG.
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
        """Rent for ``sq``.  Applies full-group doubling if ``owner``
        holds every property in the colour/railroad group."""
        group = sq.get("group")
        if group and group in GROUPS:
            group_pids = GROUPS[group]
            if all(state.properties.get(p) == owner for p in group_pids):
                return int(sq["full_rent"])
        return int(sq["rent"])

    def _pay(
        self,
        state: MLState,
        payer: str,
        creditor: Optional[str],
        amount: int,
    ) -> None:
        """Attempt to pay ``amount`` from ``payer`` to ``creditor``.

        ``creditor=None`` means a bank/tax payment (cash disappears).  If
        the payer's cash falls short, they go bankrupt immediately.
        """
        if state.cash.get(payer, 0) >= amount:
            state.cash[payer] = state.cash[payer] - amount
            if creditor is not None:
                state.cash[creditor] = state.cash.get(creditor, 0) + amount
            who = "bank" if creditor is None else creditor
            state.history.append(f"{payer} paid ${amount} to {who}")
        else:
            self._bankrupt(state, payer, creditor)

    def _bankrupt(
        self,
        state: MLState,
        player: str,
        creditor: Optional[str],
    ) -> None:
        """Declare ``player`` bankrupt.

        All cash and properties transfer to ``creditor``.  If ``creditor``
        is None (tax / bank), properties become unowned again and cash
        vanishes.  Sets ``state.winner`` and ``phase=terminal`` if only one
        solvent player remains.
        """
        cash = state.cash.get(player, 0)
        if creditor is not None and cash > 0:
            state.cash[creditor] = state.cash.get(creditor, 0) + cash
        state.cash[player] = 0

        for pid in list(state.properties.keys()):
            if state.properties[pid] == player:
                state.properties[pid] = creditor  # None = back to bank

        state.bankrupt.add(player)
        who = "bank" if creditor is None else creditor
        state.history.append(
            f"{player} is BANKRUPT — assets transferred to {who}"
        )

        solvent = [p for p in state.player_names if p not in state.bankrupt]
        if len(solvent) == 1:
            state.winner = solvent[0]
            state.phase = PH_TERMINAL
        elif len(solvent) == 0:
            state.phase = PH_TERMINAL  # edge case

    def _advance_turn(self, state: MLState) -> None:
        """Advance ``current_player_idx`` to the next solvent player and
        increment ``turn``.  Triggers turn-cap check."""
        n = len(state.player_names)
        idx = state.current_player_idx
        found = False
        for _ in range(n):
            idx = (idx + 1) % n
            if state.player_names[idx] not in state.bankrupt:
                found = True
                break
        if not found:
            # All bankrupt — shouldn't happen (bankruptcy resolves it first).
            state.phase = PH_TERMINAL
            return

        state.current_player_idx = idx
        state.phase = PH_ROLL
        state.turn += 1

        if state.turn >= state.turn_cap:
            solvent = [p for p in state.player_names if p not in state.bankrupt]
            if solvent:
                state.winner = max(
                    solvent, key=lambda p: self._net_worth(state, p)
                )
            state.phase = PH_TERMINAL

    # ------------------------------------------------------------------- step
    def step(self, state: MLState, action: Action) -> MLState:  # type: ignore[override]
        t = action.get("type", "")

        # ---- alliance actions (delegated to AllianceMixin) -----------------
        if t and t.startswith("alliance_"):
            actor = self.active_player(state)
            if actor >= 0:
                self.apply_alliance_action(state, actor, action)
            return state

        # ---- roll ----------------------------------------------------------
        if t == "roll":
            player_name = state.player_names[state.current_player_idx]
            roll = self._roll_dice(state)
            state.dice_roll = roll
            old_pos = state.positions[player_name]
            new_pos = (old_pos + roll) % BOARD_SIZE

            # Award GO salary if the player passes (or lands on) GO.
            if old_pos + roll >= BOARD_SIZE:
                state.cash[player_name] = state.cash.get(player_name, 0) + GO_SALARY
                label = "landed on" if new_pos == 0 else "passed"
                state.history.append(
                    f"{player_name} {label} GO, collected ${GO_SALARY}"
                )

            state.positions[player_name] = new_pos
            sq = BOARD[new_pos]
            sq_type = sq["type"]
            state.history.append(
                f"{player_name} rolled {roll} → sq {new_pos} ({sq_type})"
            )

            advance = True

            if sq_type == "go":
                pass  # salary already applied above
            elif sq_type == "tax":
                self._pay(state, player_name, None, sq["amount"])
                # _pay may have called _bankrupt which sets phase=terminal.
            elif sq_type in ("jail_visit", "free_parking"):
                pass  # nothing to do
            elif sq_type in ("property", "railroad"):
                pid = sq["prop_id"]
                owner = state.properties.get(pid)
                if owner is None:
                    # Unowned — offer buy decision.
                    state.pending_buy_square = new_pos
                    state.phase = PH_BUY
                    advance = False
                elif owner == player_name:
                    pass  # own it — free landing
                else:
                    # Pay rent to owner.
                    rent = self._compute_rent(state, sq, owner)
                    self._pay(state, player_name, owner, rent)

            if advance and state.phase != PH_TERMINAL:
                self._advance_turn(state)
            return state

        # ---- buy -----------------------------------------------------------
        if t == "buy":
            player_name = state.player_names[state.current_player_idx]
            sq = BOARD[state.pending_buy_square]  # type: ignore[index]
            pid = sq["prop_id"]
            price = sq["price"]
            if state.cash.get(player_name, 0) >= price:
                state.cash[player_name] -= price
                state.properties[pid] = player_name
                state.history.append(
                    f"{player_name} bought {pid} for ${price}"
                )
            else:
                # Can't afford — treat as decline (no bankruptcy from buying).
                state.history.append(
                    f"{player_name} couldn't afford {pid} (${price}), declined"
                )
            state.pending_buy_square = None
            if state.phase != PH_TERMINAL:
                self._advance_turn(state)
            return state

        # ---- decline -------------------------------------------------------
        if t == "decline":
            player_name = state.player_names[state.current_player_idx]
            if state.pending_buy_square is not None:
                sq = BOARD[state.pending_buy_square]
                state.history.append(
                    f"{player_name} declined to buy {sq.get('prop_id')}"
                )
            state.pending_buy_square = None
            if state.phase != PH_TERMINAL:
                self._advance_turn(state)
            return state

        # ---- propose_trade -------------------------------------------------
        if t == "propose_trade":
            proposer = state.player_names[state.current_player_idx]
            recipient = action.get("to", "")
            state.pending_trade = {
                "from": proposer,
                "to": recipient,
                "give_props": list(action.get("give_props", [])),
                "give_cash": int(action.get("give_cash", 0)),
                "want_props": list(action.get("want_props", [])),
                "want_cash": int(action.get("want_cash", 0)),
            }
            state.history.append(
                f"{proposer} proposed trade to {recipient}"
            )
            state.phase = PH_TRADE
            return state

        # ---- accept_trade --------------------------------------------------
        if t == "accept_trade":
            trade = state.pending_trade
            if trade is None:
                return state
            proposer = trade["from"]
            recipient = trade["to"]

            # Proposer gives give_props + give_cash to recipient.
            for pid in trade.get("give_props", []):
                if state.properties.get(pid) == proposer:
                    state.properties[pid] = recipient
            give_cash = int(trade.get("give_cash", 0))
            if give_cash > 0:
                transferred = min(give_cash, state.cash.get(proposer, 0))
                state.cash[proposer] = state.cash.get(proposer, 0) - transferred
                state.cash[recipient] = state.cash.get(recipient, 0) + transferred

            # Recipient gives want_props + want_cash to proposer.
            for pid in trade.get("want_props", []):
                if state.properties.get(pid) == recipient:
                    state.properties[pid] = proposer
            want_cash = int(trade.get("want_cash", 0))
            if want_cash > 0:
                transferred = min(want_cash, state.cash.get(recipient, 0))
                state.cash[recipient] = state.cash.get(recipient, 0) - transferred
                state.cash[proposer] = state.cash.get(proposer, 0) + transferred

            state.history.append(
                f"Trade accepted: {proposer} ↔ {recipient}"
            )
            state.pending_trade = None
            state.phase = PH_ROLL  # proposer still needs to roll
            return state

        # ---- reject_trade --------------------------------------------------
        if t == "reject_trade":
            trade = state.pending_trade
            if trade is not None:
                state.history.append(
                    f"Trade rejected by {trade['to']}"
                )
            state.pending_trade = None
            state.phase = PH_ROLL  # proposer still needs to roll
            return state

        # Ignore unknown actions.
        return state

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: MLState, player: int) -> str:  # type: ignore[override]
        player_name = state.player_names[player]
        owned = sorted(
            pid for pid, owner in state.properties.items() if owner == player_name
        )
        other_wealth = {
            state.player_names[i]: self._net_worth(state, state.player_names[i])
            for i in range(self.n_players)
            if state.player_names[i] not in state.bankrupt
            and i != player
        }
        active_alliances = [
            al for al in state.alli.alliances.values() if al.status == "active"
            and player in al.members
        ]
        alli_desc = ", ".join(
            f"#{al.id} {{{','.join('P'+str(m) for m in al.members)}}} {al.kind}"
            for al in active_alliances
        ) or "(none)"
        transcript = self.render_message_log(state, player)
        recent = "\n".join(f"  - {h}" for h in state.history[-6:]) or "  (none)"

        lines = [
            f"=== Monopoly Lite ===",
            f"You are {player_name} (seat {player}). Turn {state.turn}/{state.turn_cap}. Phase: {state.phase}.",
            f"Your cash: ${state.cash.get(player_name, 0)}  "
            f"Net worth: ${self._net_worth(state, player_name)}",
            f"Your position: sq {state.positions.get(player_name, 0)}",
            f"Your properties: {owned or '(none)'}",
            f"Bankrupt players: {sorted(state.bankrupt) or '(none)'}",
            f"Other net worths: {other_wealth}",
            f"Your alliances: {alli_desc}",
            f"Recent events:\n{recent}",
        ]
        if transcript:
            lines.append(f"Messages:\n{transcript}")

        if state.phase == PH_ROLL:
            lines.append("Action: roll the dice.")
        elif state.phase == PH_BUY and state.pending_buy_square is not None:
            sq = BOARD[state.pending_buy_square]
            lines.append(
                f"You landed on {sq.get('prop_id')} (group {sq.get('group', 'n/a')}, "
                f"price ${sq['price']}, rent ${sq['rent']}).  Buy or decline?"
            )
        elif state.phase == PH_TRADE and state.pending_trade:
            tr = state.pending_trade
            lines.append(
                f"Trade offer from {tr['from']}: "
                f"they give props={tr['give_props']} cash=${tr['give_cash']}; "
                f"they want props={tr['want_props']} cash=${tr['want_cash']}. "
                "Accept or reject?"
            )

        return "\n".join(lines)

    def parse_action(self, state: MLState, player: int, text: str) -> Action:  # type: ignore[override]
        phase = state.phase
        if phase == PH_ROLL:
            return {"type": "roll"}
        if phase == PH_BUY:
            low = text.lower()
            if "buy" in low and "decline" not in low:
                return {"type": "buy"}
            return {"type": "decline"}
        if phase == PH_TRADE:
            if "accept" in text.lower():
                return {"type": "accept_trade"}
            return {"type": "reject_trade"}
        raise ParseError(f"no parseable action in phase {phase!r}")

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
            here = [
                name for name, p in state.positions.items() if p == pos
            ]
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
                    f"{p}=${state.cash.get(p, 0)}"
                    for p in state.player_names
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
        t = action.get("type", "")
        if t and t.startswith("alliance_"):
            return self.alliance_observations(new_state, action, actor)
        return [
            Obs(
                audience=list(range(self.n_players)),
                payload={"type": "action", "player": actor, "action": action},
            )
        ]

    # ------------------------------------------------------ alliance hooks
    def alliance_legal_actions(self, state: MLState, player: int) -> List[Action]:
        """Nonaggression pacts available to non-bankrupt seats."""
        alli = state.alli
        actions: List[Action] = []
        for al in alli.alliances.values():
            if al.status == "proposed" and player in al.pending:
                actions.append({"type": "alliance_accept", "alliance_id": al.id})
                actions.append({"type": "alliance_decline", "alliance_id": al.id})
            if al.status == "active" and player in al.members:
                actions.append(
                    {"type": "alliance_break", "alliance_id": al.id, "reason": ""}
                )
        living = self.living_seats(state)
        for other in living:
            if other != player:
                actions.append(
                    {"type": "alliance_propose", "to": [other],
                     "kind": "nonaggression", "terms": {}}
                )
        return actions

    def judge_alliance(
        self, state: MLState, action: Action, actor: int
    ) -> List[dict]:
        """Monopoly Lite does not mechanically enforce alliance honour/betray.
        Observable via the AllianceState audit trail (``metrics`` layer)."""
        return []
