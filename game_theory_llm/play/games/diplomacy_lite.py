"""Diplomacy — standard 7-power Europe, Full-Press, with documented lite
simplifications (spec §5.4 / §9; ``config.adjudicator == "lite-press-v2"``).

This module replaces the v0 6-territory toy with the real standard map driven
by ``game_theory_llm.play.maps.diplomacy_map`` (75 provinces, 34 supply
centres, 7 powers, army+fleet adjacency graphs, 1901 opening).

Phase engine (per game-year)::

    Spring move  (PH_SPRING)
      -> Spring retreat  (PH_RETREAT)   [only if dislodgements occurred]
    Fall move    (PH_FALL)
      -> Fall retreat    (PH_RETREAT)   [only if dislodgements occurred]
    Winter build/disband (PH_WINTER)    [only if any power's count != SC count]

A Full-Press ``PH_NEGOTIATION`` runs *before* each movement phase (Spring and
Fall) for ``config.nego_rounds`` rounds, carrying public ``say`` + ``whisper``
and the alliance kinds ``support_pact`` / ``dmz`` / ``nonaggression`` /
``coalition``.

Adjudication is isolated behind the PURE function ``adjudicate(orders, board)
-> Resolution`` so it can be unit-tested against a DATC-style subset
independently of the messaging / alliance / observation layer
(``tests/play/test_diplomacy.py``).

Honour / betray (judged at Fall resolution, where SC counts settle, plus at
every movement resolution for support pacts)::

    honored : a promised SUPPORT order for an ally actually appears in the
              orders this power submitted.
    betrayed: a MOVE into an active ally's occupied province or owned SC, OR
              omitting a support order the alliance terms promised.

Betrayal is always physically possible (cheap-talk alliances; no hard
enforcement), so the steering signal is measurable.

================================================================================
LITE SIMPLIFICATIONS (honestly flagged; ``config.adjudicator='lite-press-v2'``)
================================================================================
* **Single-fleet convoys only.** A convoy uses exactly one fleet on a sea
  province adjacent to both the army's origin and destination coasts. Convoy
  *chains* (2+ fleets) raise ``ParseError`` at parse time and are never
  adjudicated.
* **Convoy paradoxes via the Szykman rule.** Any convoy whose convoying fleet
  is dislodged, or which forms a paradox, is treated as a FAILED convoy (the
  army holds). No fixpoint solver.
* **Split coasts modelled as single provinces.** Spain / St Petersburg /
  Bulgaria are single provinces (see ``diplomacy_map``); ``COASTS == {}``.
* **Deterministic retreats.** A dislodged unit retreats to the lowest-index
  legal, empty, non-contested, non-attacker-origin province; if none exists it
  disbands. No retreat standoffs.
* **Builds on empty owned home SC only; excess builds waived.** A power builds
  up to (SC count - unit count) units on its empty home supply centres in
  lowest-province-index order; if it cannot place them all (home SCs occupied)
  the excess builds are simply waived.
* Beleaguered-garrison and self-standoff nuances resolve by the deterministic
  strength comparison below (documented, not separately special-cased).

================================================================================
HARD YEAR CAP
================================================================================
The match is capped at ``MAX_YEARS`` game-years (default 5). At the cap the
winner is the power with the most supply centres (ties broken by lowest seat
index). A solo victory (>=18 SCs) ends the game immediately. Under 7
``RandomPlayer`` seats with the default messaging-on config the match
terminates well within ``max_turns=200``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..alliances import AllianceMixin, AllianceState, new_event
from ..base import Action, Game, GOD, Obs, ParseError
from ..config import GameConfig
from ..maps import diplomacy_map as M
from ..messaging import MessagingMixin, NegotiationState

# ---------------------------------------------------------------------------
# Powers / seats
# ---------------------------------------------------------------------------
POWERS: List[str] = [
    "Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey",
]
N_PLAYERS = len(POWERS)
SOLO_SC = 18           # supply centres for an instant solo win
MAX_YEARS = 5          # hard cap (game-years); winner = most SCs at cap

# Phase tags
PH_NEGOTIATION = "negotiation"
PH_SPRING = "spring_move"
PH_FALL = "fall_move"
PH_RETREAT = "retreat"
PH_WINTER = "winter"
PH_TERMINAL = "terminal"

ALLIANCE_KINDS = ("support_pact", "dmz", "nonaggression", "coalition")

# Adjacency lookups (id -> frozenset[id]) from the map module.
ADJ_ARMY = M.ADJ_ARMY
ADJ_FLEET = M.ADJ_FLEET
KIND = M.KIND
PROV = M.PROVINCES
SUPPLY_CENTRE = M.SUPPLY_CENTRE
HOME_SC = M.HOME_SC


def _abbr(pid: int) -> str:
    return PROV[pid]


def _adj_for(unit_type: str, pid: int):
    return ADJ_FLEET[pid] if unit_type == "F" else ADJ_ARMY[pid]


def _can_occupy(unit_type: str, pid: int) -> bool:
    k = KIND[pid]
    if unit_type == "A":
        return k in ("inland", "coastal")
    return k in ("sea", "coastal")


# ===========================================================================
# PURE ADJUDICATION (no game-state coupling; tested directly).
# ===========================================================================
@dataclass
class Resolution:
    """Outcome of one movement phase.

    Attributes
    ----------
    moves : dict[int, int]
        For each successfully moving unit, ``origin -> destination``.
    holds : set[int]
        Origins of units that did not move (held, supported, or bounced).
    dislodged : dict[int, dict]
        ``province -> {"power", "type", "from"}`` for each dislodged unit.
        ``from`` is the attacker origin (forbidden retreat target).
    bounces : set[int]
        Destination provinces where a standoff occurred (no one entered).
    cut_supports : set[int]
        Origins whose SUPPORT order was cut.
    failed_convoys : set[int]
        Origins of convoyed armies whose convoy failed (treated as hold).
    contested : set[int]
        Destination provinces that had >=1 incoming move attempt (used for
        retreat legality — a unit may not retreat to a contested space).
    """

    moves: Dict[int, int] = field(default_factory=dict)
    holds: set = field(default_factory=set)
    dislodged: Dict[int, dict] = field(default_factory=dict)
    bounces: set = field(default_factory=set)
    cut_supports: set = field(default_factory=set)
    failed_convoys: set = field(default_factory=set)
    contested: set = field(default_factory=set)


@dataclass
class Board:
    """Immutable-ish board snapshot passed to ``adjudicate``.

    ``units[pid] = (unit_type, power_name)``. Adjudication never mutates the
    Board; it returns a ``Resolution`` the caller applies.
    """

    units: Dict[int, Tuple[str, str]] = field(default_factory=dict)

    def power_of(self, pid: int) -> Optional[str]:
        u = self.units.get(pid)
        return u[1] if u else None

    def type_of(self, pid: int) -> Optional[str]:
        u = self.units.get(pid)
        return u[0] if u else None


def _convoy_ok(army_src: int, dst: int, fleet_prov: int, board: Board,
               orders: Dict[int, dict]) -> bool:
    """Single-fleet convoy validity (Szykman-lite).

    The convoying fleet must (a) be a fleet on a sea province, (b) be
    fleet-adjacent to BOTH the army's origin and destination, (c) actually
    issue a matching CONVOY order, and (d) survive (not be dislodged — handled
    by the caller marking failed convoys). Convoy chains are rejected at parse
    time, so only one fleet is ever considered here.
    """
    if board.type_of(fleet_prov) != "F":
        return False
    if KIND[fleet_prov] != "sea":
        return False
    if army_src not in ADJ_FLEET[fleet_prov]:
        return False
    if dst not in ADJ_FLEET[fleet_prov]:
        return False
    fo = orders.get(fleet_prov)
    if not fo or fo.get("type") != "CONVOY":
        return False
    return fo.get("army") == army_src and fo.get("target") == dst


def adjudicate(orders: Dict[int, dict], board: Board) -> Resolution:
    """Resolve one movement phase. PURE: depends only on ``orders`` + ``board``.

    ``orders`` maps an occupied province id to an order dict::

        {"type": "HOLD"}
        {"type": "MOVE", "target": dst, "via_convoy": bool}
        {"type": "SUPPORT", "target": tgt}                      # support-hold
        {"type": "SUPPORT", "attacker": src, "target": dst}     # support-move
        {"type": "CONVOY", "army": a_src, "target": dst}        # fleet convoys army

    Units with no order default to HOLD. Returns a ``Resolution``.

    Resolution is a FIXPOINT iteration (standard Diplomacy adjudication, not a
    single pass). On each pass we:

      1. recompute which supports are cut (a support is cut by any attacker on
         the supporter's province that is not the unit being supported against,
         provided that attacker is not itself failing solely because the
         supporter's own move-support cancels it — we use the conservative rule
         that ANY foreign attack with strength >= 1 cuts);
      2. recompute MOVE / HOLD strengths, where a unit's HOLD strength is full
         (``1 + valid support-holds``) UNLESS its own move SUCCEEDS this pass
         (B1: a *bounced* mover defends its origin at full strength — it only
         contributes 0 to its origin's defence if its move actually succeeds);
      3. resolve every contested province, applying the self-dislodgement ban
         (B2: a power's support cannot help dislodge that power's own unit, and
         the occupant counts as "vacating" only if its move SUCCEEDS), and the
         head-to-head swap rule;
      4. mark dislodgements; then if any convoying fleet was dislodged, fail
         that convoy (B4 / Szykman) so the convoyed army holds — and re-iterate.

    The loop repeats until the set of successful moves and failed convoys
    stabilises (bounded; each pass can only turn moves from success->failure or
    convoys from ok->failed, so it monotonically converges).
    """
    res = Resolution()

    # Default missing orders to HOLD.
    full: Dict[int, dict] = {}
    for pid in board.units:
        full[pid] = orders.get(pid, {"type": "HOLD"})

    # Static (pass-invariant) move destinations from the submitted orders.
    base_move_dst: Dict[int, int] = {}
    for src, od in full.items():
        if od.get("type") == "MOVE":
            base_move_dst[src] = od["target"]

    # Convoys that have failed (structurally invalid OR dislodged convoyer).
    # Iteratively grown across passes (B4). Once failed, a convoyed army holds.
    failed_convoys: set = set()

    def _convoy_fleet(src: int, dst: int) -> Optional[int]:
        """Return the single convoying fleet for army ``src`` -> ``dst`` if a
        structurally valid CONVOY order exists, else None."""
        for fp, fo in full.items():
            if fo.get("type") == "CONVOY" and fo.get("army") == src \
                    and fo.get("target") == dst:
                if _convoy_ok(src, dst, fp, board, full):
                    return fp
                return None
        return None

    def _resolve_pass(failed: set):
        """One adjudication pass. Returns
        ``(move_success, support_valid, winners, by_dst, convoy_move)``.

        ``failed`` is the set of convoyed armies whose convoy has failed so far
        (they are treated as HOLD this pass).
        """
        # Effective move destinations this pass: drop failed convoys.
        move_dst: Dict[int, int] = {
            s: d for s, d in base_move_dst.items() if s not in failed
        }
        convoy_move: Dict[int, int] = {}
        for s in list(move_dst):
            od = full[s]
            if od.get("via_convoy"):
                fleet = _convoy_fleet(s, move_dst[s])
                if fleet is None:
                    # Structurally invalid convoy -> failed (caller records it).
                    failed.add(s)
                    move_dst.pop(s, None)
                else:
                    convoy_move[s] = fleet

        attacks_into: Dict[int, List[int]] = {}
        for s, d in move_dst.items():
            attacks_into.setdefault(d, []).append(s)

        # ---- support cuts ----
        support_valid: Dict[int, bool] = {}
        for sp, od in full.items():
            if od.get("type") != "SUPPORT":
                continue
            supported_into = od["target"]
            cut = False
            for atk in attacks_into.get(sp, []):
                if atk == supported_into:
                    # An attack from the very province being supported against
                    # does not cut the support (standard rule): you cannot cut a
                    # support directed against your own attack by making it.
                    continue
                # Any OTHER attacker on the supporter's province cuts the
                # support (spec subset: "support-cut by any non-supported
                # attacker"). This includes same-power attackers — the only
                # protection against helping a self-dislodgement is the
                # self-dislodge ban applied during resolution below.
                cut = True
                break
            support_valid[sp] = not cut

        # ---- strengths (depend on move_success for hold strength) ----
        def move_strength(src: int, dst: int) -> int:
            s = 1
            for sp, sod in full.items():
                if sod.get("type") == "SUPPORT" \
                        and sod.get("attacker") == src \
                        and sod.get("target") == dst \
                        and support_valid.get(sp, False):
                    s += 1
            return s

        def hold_strength(pid: int, move_success: Dict[int, bool]) -> int:
            if pid not in board.units:
                return 0
            # B1: a mover contributes 0 to its origin's defence ONLY if its
            # move SUCCEEDS. A bounced / failed / failed-convoy mover defends
            # its origin at full strength.
            if full.get(pid, {}).get("type") == "MOVE" \
                    and pid not in failed and move_success.get(pid, False):
                return 0
            s = 1
            for sp, sod in full.items():
                if sod.get("type") == "SUPPORT" and "attacker" not in sod \
                        and sod.get("target") == pid \
                        and support_valid.get(sp, False):
                    s += 1
            return s

        by_dst: Dict[int, List[int]] = {}
        for s, d in move_dst.items():
            by_dst.setdefault(d, []).append(s)

        # Inner fixpoint over move_success (hold strength depends on whether
        # the occupant successfully vacates, which depends on its own move).
        move_success: Dict[int, bool] = {s: True for s in move_dst}
        winners: Dict[int, int] = {}
        for _ in range(len(move_dst) + 2):
            winners = {}
            new_success: Dict[int, bool] = {s: False for s in move_dst}
            for dst, srcs in by_dst.items():
                strengths = {s: move_strength(s, dst) for s in srcs}
                best = max(strengths.values())
                top = [s for s in srcs if strengths[s] == best]
                if len(top) > 1:
                    continue  # standoff -> bounce; nobody enters
                winner = top[0]
                occ_order = full.get(dst)
                occupant_present = dst in board.units

                # Head-to-head: winner -> dst while occupant -> winner's origin
                # (neither convoyed). Resolve by strength; ties bounce.
                head_to_head = (occupant_present and occ_order is not None
                                and occ_order.get("type") == "MOVE"
                                and occ_order.get("target") == winner
                                and winner not in convoy_move
                                and dst not in convoy_move)
                if head_to_head:
                    ws = move_strength(winner, dst)
                    os_ = move_strength(dst, winner)
                    # Self-dislodge ban also applies to head-to-head.
                    if ws > os_ and not (
                            board.power_of(dst) == board.power_of(winner)):
                        winners[dst] = winner
                        new_success[winner] = True
                    continue

                if not occupant_present:
                    winners[dst] = winner
                    new_success[winner] = True
                    continue

                # Occupant present and not head-to-head.
                ws = move_strength(winner, dst)
                # The occupant vacates only if its own move SUCCEEDS (B2).
                occ_vacates = (occ_order is not None
                               and occ_order.get("type") == "MOVE"
                               and move_success.get(dst, False))
                if occ_vacates:
                    # Square is being vacated: winner needs only to beat any
                    # competing movers (already the unique top) -> enters.
                    winners[dst] = winner
                    new_success[winner] = True
                    continue
                defender = hold_strength(dst, move_success)
                if ws > defender:
                    occ_power = board.power_of(dst)
                    win_power = board.power_of(winner)
                    # B2: self-dislodgement ban — a power may not dislodge its
                    # own (non-vacating) unit, even with its own support.
                    if occ_power == win_power:
                        continue  # bounce
                    winners[dst] = winner
                    new_success[winner] = True
                # else: bounce (defender holds).
            if new_success == move_success:
                move_success = new_success
                break
            move_success = new_success

        return move_success, support_valid, winners, by_dst, convoy_move

    # ---- Outer fixpoint over failed convoys (B4 / Szykman). ----------------
    move_success: Dict[int, bool] = {}
    support_valid: Dict[int, bool] = {}
    winners: Dict[int, int] = {}
    by_dst: Dict[int, List[int]] = {}
    convoy_move: Dict[int, int] = {}
    for _ in range(len(board.units) + 2):
        before = set(failed_convoys)
        move_success, support_valid, winners, by_dst, convoy_move = \
            _resolve_pass(failed_convoys)

        # Compute dislodgements implied by this pass to detect dislodged
        # convoying fleets (B4): if a convoying fleet is dislodged, its convoy
        # fails and the convoyed army must hold; re-iterate.
        # ``winners`` maps dst -> winning src, so a unit moved OUT iff it is a
        # winning src (``move_success[pid]``), and is dislodged iff someone
        # won the move INTO its province while it did not move out.
        moved_out = {s for s in move_success if move_success[s]}
        dislodged_now: set = set()
        for pid in board.units:
            if pid in moved_out:         # this unit successfully moved out
                continue
            incoming = winners.get(pid)
            if incoming is not None and incoming != pid:
                dislodged_now.add(pid)

        grew = False
        for army, fleet in convoy_move.items():
            if fleet in dislodged_now and army not in failed_convoys:
                failed_convoys.add(army)
                grew = True
        if not grew and failed_convoys == before:
            break

    # ---- Materialise the Resolution from the final pass. -------------------
    for dst, src in winners.items():
        res.moves[src] = dst
    res.failed_convoys = set(failed_convoys)
    res.cut_supports = {sp for sp, ok in support_valid.items() if not ok}

    # Contested = any province that received >=1 move attempt this final pass.
    for dst in by_dst:
        res.contested.add(dst)
    # Bounces = contested destinations no one successfully entered.
    # ``winners`` maps dst -> winning src; a dst absent from it had a standoff.
    for dst in res.contested:
        if dst not in winners:
            res.bounces.add(dst)

    # Dislodgements + holds.
    for pid, (utype, power) in board.units.items():
        if pid in res.moves:             # successfully moved out
            continue
        incoming = winners.get(pid)
        if incoming is not None and incoming != pid:
            res.dislodged[pid] = {
                "power": power, "type": utype, "from": incoming,
            }
        else:
            res.holds.add(pid)

    return res


# ===========================================================================
# GAME STATE
# ===========================================================================
@dataclass
class DipState:
    n_players: int = N_PLAYERS
    # units[pid] = (type, power_name)
    units: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    # sc_owner[pid] = power_name  (only supply centres tracked; persists)
    sc_owner: Dict[int, str] = field(default_factory=dict)
    # Pending orders for the current movement phase: pid -> order dict
    pending_orders: Dict[int, dict] = field(default_factory=dict)
    orders_done: List[bool] = field(default_factory=list)   # per seat
    # Retreat bookkeeping: pid -> {"power","type","from"}; resolved in PH_RETREAT
    dislodged: Dict[int, dict] = field(default_factory=dict)
    contested: set = field(default_factory=set)             # provs that bounced (retreat-illegal)
    retreats_done: List[bool] = field(default_factory=list)
    # Winter build/disband bookkeeping
    builds_done: List[bool] = field(default_factory=list)
    # Season tracking
    year: int = 1901
    season: str = "Spring"     # "Spring" | "Fall"
    phase: str = PH_NEGOTIATION
    history: List[str] = field(default_factory=list)
    winner: Optional[int] = None         # winning seat idx
    win_reason: str = ""
    turn: int = 0
    # Messaging / alliances
    nego: NegotiationState = field(default_factory=NegotiationState)
    alli: AllianceState = field(default_factory=AllianceState)
    # Per-power record of orders this movement phase, for honour/betray judging
    last_orders_by_power: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    # Public resolution payload, set when a movement phase resolves. Consumed
    # ONCE by ``observations`` to emit the single post-resolution public Obs
    # (B5: per-power orders stay private until the whole phase resolves).
    last_resolution: Optional[dict] = None


# ===========================================================================
# THE GAME
# ===========================================================================
class DiplomacyLite(MessagingMixin, AllianceMixin, Game):
    name = "diplomacy"
    n_players = N_PLAYERS

    def __init__(self, config: "Optional[GameConfig]" = None) -> None:
        super().__init__(config)
        # Document the adjudicator variant for the match_start.config record.
        self.adjudicator = "lite-press-v2"

    # ----------------------------------------------------------- power/seat
    @staticmethod
    def power_name(seat: int) -> str:
        return POWERS[seat]

    @staticmethod
    def seat_of(power: str) -> int:
        return POWERS.index(power)

    # ----------------------------------------------------------- setup
    def initial_state(self, rng) -> DipState:
        units: Dict[int, Tuple[str, str]] = {}
        sc_owner: Dict[int, str] = {}
        for power, ulist in M.START_UNITS.items():
            for utype, pid, _coast in ulist:
                units[pid] = (utype, power)
            for pid in HOME_SC[power]:
                sc_owner[pid] = power
        st = DipState(
            n_players=self.n_players,
            units=units,
            sc_owner=sc_owner,
            orders_done=[False] * self.n_players,
            retreats_done=[False] * self.n_players,
            builds_done=[False] * self.n_players,
            year=1901,
            season="Spring",
            phase=PH_NEGOTIATION,
        )
        # Spring negotiation precedes the first Spring movement phase.
        if self.config.messaging:
            self.start_negotiation(st, return_phase=PH_SPRING,
                                   rounds=self.config.nego_rounds)
        else:
            st.phase = PH_SPRING
        return st

    # ----------------------------------------------------------- helpers
    def _units_of(self, state: DipState, power: str) -> List[int]:
        return [p for p, (t, o) in state.units.items() if o == power]

    def _sc_count(self, state: DipState, power: str) -> int:
        return sum(1 for o in state.sc_owner.values() if o == power)

    def _alive(self, state: DipState, seat: int) -> bool:
        """A power is alive if it owns >=1 unit OR >=1 supply centre."""
        power = POWERS[seat]
        return bool(self._units_of(state, power)) or self._sc_count(state, power) > 0

    def living_seats(self, state: DipState) -> List[int]:
        return [s for s in range(self.n_players) if self._alive(state, s)]

    # ----------------------------------------------------- active_player
    def active_player(self, state: DipState) -> int:
        if state.phase == PH_TERMINAL:
            return -1
        if state.phase == PH_NEGOTIATION:
            return self.nego_active_player(state)
        if state.phase in (PH_SPRING, PH_FALL):
            for s in range(self.n_players):
                if state.orders_done[s]:
                    continue
                if self._units_of(state, POWERS[s]):
                    return s
            return -1
        if state.phase == PH_RETREAT:
            for s in range(self.n_players):
                if state.retreats_done[s]:
                    continue
                if self._dislodged_of(state, POWERS[s]):
                    return s
            return -1
        if state.phase == PH_WINTER:
            for s in range(self.n_players):
                if state.builds_done[s]:
                    continue
                if self._build_delta(state, POWERS[s]) != 0:
                    return s
            return -1
        return -1

    def _dislodged_of(self, state: DipState, power: str) -> List[int]:
        return [p for p, d in state.dislodged.items() if d["power"] == power]

    def _build_delta(self, state: DipState, power: str) -> int:
        """+n => may build n; -n => must disband n; 0 => nothing."""
        return self._sc_count(state, power) - len(self._units_of(state, power))

    # ----------------------------------------------------- legal_actions
    def legal_actions(self, state: DipState, player: int) -> List[Action]:
        if state.phase == PH_NEGOTIATION:
            return self.nego_legal_actions(state, player)
        if state.phase in (PH_SPRING, PH_FALL):
            return self._movement_legal_actions(state, player)
        if state.phase == PH_RETREAT:
            return self._retreat_legal_actions(state, player)
        if state.phase == PH_WINTER:
            return self._winter_legal_actions(state, player)
        return []

    def _movement_legal_actions(self, state: DipState, player: int) -> List[Action]:
        """Return a curated set of COMPLETE order sets (one order per unit).

        We do not enumerate the cross-product (exponential). We offer: all-hold,
        and for each unit a "this unit moves to its lowest legal neighbour,
        others hold". RandomPlayer picks among these; LLMs use ``parse_action``.
        """
        power = POWERS[player]
        units = sorted(self._units_of(state, power))
        if not units:
            return [{"type": "submit_orders", "orders": {}}]
        actions: List[Action] = []
        all_hold = {u: {"type": "HOLD"} for u in units}
        actions.append({"type": "submit_orders", "orders": dict(all_hold)})
        for u in units:
            utype = state.units[u][0]
            for nb in sorted(_adj_for(utype, u)):
                if not _can_occupy(utype, nb):
                    continue
                orders = {v: {"type": "HOLD"} for v in units}
                orders[u] = {"type": "MOVE", "target": nb, "via_convoy": False}
                actions.append({"type": "submit_orders", "orders": orders})
        return actions

    def _retreat_legal_actions(self, state: DipState, player: int) -> List[Action]:
        power = POWERS[player]
        dis = sorted(self._dislodged_of(state, power))
        orders: Dict[int, dict] = {}
        for pid in dis:
            tgt = self._auto_retreat_target(state, pid)
            if tgt is None:
                orders[pid] = {"type": "DISBAND"}
            else:
                orders[pid] = {"type": "RETREAT", "target": tgt}
        return [{"type": "submit_retreats", "orders": orders}]

    def _winter_legal_actions(self, state: DipState, player: int) -> List[Action]:
        power = POWERS[player]
        delta = self._build_delta(state, power)
        if delta > 0:
            sites = self._buildable_sites(state, power)
            orders = {}
            for i, pid in enumerate(sites[:delta]):
                # Default build an army (fleet on coastal SC if inland-only? keep A).
                orders[pid] = {"type": "BUILD", "unit": "A"}
            return [{"type": "submit_builds", "orders": orders}]
        if delta < 0:
            n = -delta
            victims = self._auto_disband_order(state, power)[:n]
            orders = {pid: {"type": "DISBAND"} for pid in victims}
            return [{"type": "submit_builds", "orders": orders}]
        return [{"type": "submit_builds", "orders": {}}]

    def _buildable_sites(self, state: DipState, power: str) -> List[int]:
        """Empty owned home SCs, lowest-index first."""
        out = []
        for pid in sorted(HOME_SC[power]):
            if state.sc_owner.get(pid) == power and pid not in state.units:
                out.append(pid)
        return out

    def _auto_disband_order(self, state: DipState, power: str) -> List[int]:
        """Deterministic disband priority: lowest province index first."""
        return sorted(self._units_of(state, power))

    def _auto_retreat_target(self, state: DipState, pid: int) -> Optional[int]:
        """Deterministic lowest-index legal empty non-contested retreat space.

        Legal = adjacency-reachable for the unit type, not currently occupied,
        not the province the dislodging attacker came from, not a bounced
        (contested) province. Returns None -> disband.
        """
        utype = state.dislodged[pid]["type"]
        forbidden_from = state.dislodged[pid]["from"]
        for nb in sorted(_adj_for(utype, pid)):
            if not _can_occupy(utype, nb):
                continue
            if nb in state.units:
                continue
            if nb == forbidden_from:
                continue
            if nb in state.contested:
                continue
            return nb
        return None

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: DipState, player: int) -> str:
        head = self._public_header(state, player)
        chat = self.render_message_log(state, player)
        chat_block = f"\n\n=== Negotiation transcript (your view) ===\n{chat}" if chat else ""
        if state.phase == PH_NEGOTIATION:
            body = self._nego_prompt(state, player)
        elif state.phase in (PH_SPRING, PH_FALL):
            body = self._movement_prompt(state, player)
        elif state.phase == PH_RETREAT:
            body = self._retreat_prompt(state, player)
        elif state.phase == PH_WINTER:
            body = self._winter_prompt(state, player)
        else:
            body = "[Game over.]"
        return f"{head}{chat_block}\n\n{body}"

    def _public_header(self, state: DipState, player: int) -> str:
        power = POWERS[player]
        my_units = [f"{state.units[p][0]} {_abbr(p)}"
                    for p in sorted(self._units_of(state, power))]
        sc_line = " ".join(
            f"{POWERS[s][:3]}={self._sc_count(state, POWERS[s])}"
            for s in range(self.n_players)
        )
        return (
            "=== Diplomacy (standard map, Full-Press, lite-press-v2) ===\n"
            f"You are {power} (seat P{player}). "
            f"{state.season} {state.year}. Phase: {state.phase}.\n"
            f"Your units: {my_units}\n"
            f"Supply centres: {sc_line}\n"
            f"Active alliances: {self._alliance_brief(state, player)}"
        )

    def _alliance_brief(self, state: DipState, player: int) -> str:
        parts = []
        for al in state.alli.alliances.values():
            if al.status == "active" and player in al.members:
                others = ",".join(f"P{m}" for m in al.members if m != player)
                parts.append(f"#{al.id}[{al.kind} with {others}]")
        return " ".join(parts) if parts else "(none)"

    def _nego_prompt(self, state: DipState, player: int) -> str:
        return (
            "Negotiation phase (Full-Press). You may send messages and form "
            "alliances before orders are written. One action per turn:\n"
            "  <say>public message</say>\n"
            "  <whisper to=2,4>private message</whisper>\n"
            "  <pass></pass>   (say nothing this slot)\n"
            "Alliances (cheap talk — you CAN later betray them):\n"
            "  <ally propose to=2 kind=support_pact>I'll support your move</ally>\n"
            f"     kind in {ALLIANCE_KINDS}\n"
            "  <ally accept ID>  <ally decline ID>  <ally break ID>reason</ally>"
        )

    def _movement_prompt(self, state: DipState, player: int) -> str:
        power = POWERS[player]
        units = sorted(self._units_of(state, power))
        lines = []
        for u in units:
            utype = state.units[u][0]
            nbrs = sorted(n for n in _adj_for(utype, u) if _can_occupy(utype, n))
            nb_str = ", ".join(_abbr(n) for n in nbrs)
            lines.append(f"  {utype} {_abbr(u)} -> can move to: {nb_str}")
        return (
            f"{state.season} movement orders. Submit ONE order per unit you "
            f"own ({[_abbr(u) for u in units]}).\n"
            "Order syntax (province 3-letter codes, case-insensitive):\n"
            "  A PAR HOLD\n"
            "  A PAR - BUR                (move)\n"
            "  A PAR S A MAR - BUR        (support a move)\n"
            "  A PAR S F BRE              (support a hold)\n"
            "  A LON - BEL VIA CONVOY     (single-fleet convoy)\n"
            "  F ENG C A LON - BEL        (fleet convoys the army)\n"
            "Format your full set inside one block:\n"
            "<orders>\n" + "\n".join(lines) + "\n</orders>"
        )

    def _retreat_prompt(self, state: DipState, player: int) -> str:
        power = POWERS[player]
        dis = sorted(self._dislodged_of(state, power))
        lines = []
        for pid in dis:
            tgt = self._auto_retreat_target(state, pid)
            opt = (f"retreat to {_abbr(tgt)} (auto if you do nothing)"
                   if tgt is not None else "must DISBAND (no legal retreat)")
            lines.append(f"  {state.dislodged[pid]['type']} {_abbr(pid)}: {opt}")
        return (
            "Retreat phase. Your dislodged units (deterministic auto-retreat "
            "applies if you submit nothing):\n" + "\n".join(lines) + "\n"
            "<retreats>\n  F PRU - BAL\n  A SIL DISBAND\n</retreats>"
        )

    def _winter_prompt(self, state: DipState, player: int) -> str:
        power = POWERS[player]
        delta = self._build_delta(state, power)
        if delta > 0:
            sites = self._buildable_sites(state, power)
            return (
                f"Winter adjustment: you may BUILD {delta} unit(s) on empty "
                f"home centres {[_abbr(s) for s in sites]} (excess waived).\n"
                "<builds>\n  BUILD A PAR\n  BUILD F BRE\n</builds>"
            )
        if delta < 0:
            return (
                f"Winter adjustment: you must DISBAND {-delta} unit(s) "
                f"(lowest-index auto if you submit nothing).\n"
                "<builds>\n  DISBAND A MUN\n</builds>"
            )
        return "Winter adjustment: no change required. Submit <builds></builds>."

    # -------------------------------------------------------------- parse
    def parse_action(self, state: DipState, player: int, text: str) -> Action:
        if state.phase == PH_NEGOTIATION:
            return self.nego_parse(state, player, text)
        if state.phase in (PH_SPRING, PH_FALL):
            return self._parse_movement(state, player, text)
        if state.phase == PH_RETREAT:
            return self._parse_retreats(state, player, text)
        if state.phase == PH_WINTER:
            return self._parse_builds(state, player, text)
        raise ParseError(f"no action expected in phase {state.phase}")

    _BLOCK_RE = re.compile(r"<orders>(.*?)</orders>", re.I | re.DOTALL)
    _RET_BLOCK_RE = re.compile(r"<retreats>(.*?)</retreats>", re.I | re.DOTALL)
    _BUILD_BLOCK_RE = re.compile(r"<builds>(.*?)</builds>", re.I | re.DOTALL)

    def _pid_from_token(self, tok: str) -> int:
        tok = tok.strip().upper()
        if tok not in M._ID:
            raise ParseError(f"unknown province {tok!r}")
        return M._ID[tok]

    def _parse_movement(self, state: DipState, player: int, text: str) -> Action:
        m = self._BLOCK_RE.search(text)
        if not m:
            raise ParseError("expected an <orders>...</orders> block")
        power = POWERS[player]
        my_units = set(self._units_of(state, power))
        orders: Dict[int, dict] = {}
        for raw in m.group(1).splitlines():
            line = raw.strip()
            if not line:
                continue
            order = self._parse_one_order(state, player, my_units, line)
            if order is not None:
                src, od = order
                orders[src] = od
        for u in my_units:
            orders.setdefault(u, {"type": "HOLD"})
        # §9 / docstring guarantee: multi-fleet convoy chains are disallowed and
        # raise a ParseError at parse time (never adjudicated). A chain is any
        # convoyed army for which two or more CONVOY orders are issued (more
        # than one convoying fleet would be required to carry it).
        convoy_fleets: Dict[Tuple[int, int], int] = {}
        for od in orders.values():
            if od.get("type") == "CONVOY":
                key = (od.get("army"), od.get("target"))
                convoy_fleets[key] = convoy_fleets.get(key, 0) + 1
        for (army, dst), count in convoy_fleets.items():
            if count >= 2:
                raise ParseError(
                    f"multi-fleet convoy chains are disallowed "
                    f"(lite-press-v2): {count} fleets ordered to convoy "
                    f"{_abbr(army)} -> {_abbr(dst)}; only single-fleet convoys "
                    f"are supported")
        return {"type": "submit_orders", "orders": orders}

    _UNIT_RE = re.compile(r"^([AF])\s+([A-Za-z]{3})\s+(.*)$", re.I)

    def _parse_one_order(self, state, player, my_units, line):
        m = self._UNIT_RE.match(line)
        if not m:
            raise ParseError(f"cannot parse order line: {line!r}")
        utype = m.group(1).upper()
        src = self._pid_from_token(m.group(2))
        rest = m.group(3).strip()
        if src not in my_units:
            raise ParseError(f"you do not own a unit on {_abbr(src)}")
        if state.units[src][0] != utype:
            raise ParseError(f"unit on {_abbr(src)} is a "
                             f"{state.units[src][0]}, not {utype}")
        up = rest.upper()

        # CONVOY: F ENG C A LON - BEL
        cm = re.match(r"C\s+A\s+([A-Za-z]{3})\s*-\s*([A-Za-z]{3})", rest, re.I)
        if cm:
            if utype != "F":
                raise ParseError("only a fleet may convoy")
            a_src = self._pid_from_token(cm.group(1))
            dst = self._pid_from_token(cm.group(2))
            if KIND[src] != "sea":
                raise ParseError("convoying fleet must be at sea")
            return src, {"type": "CONVOY", "army": a_src, "target": dst}

        # SUPPORT MOVE: A PAR S A MAR - BUR  /  support hold: A PAR S F BRE
        sm = re.match(r"S\s+[AF]?\s*([A-Za-z]{3})\s*-\s*([A-Za-z]{3})", rest, re.I)
        if sm:
            atk = self._pid_from_token(sm.group(1))
            dst = self._pid_from_token(sm.group(2))
            if dst not in _adj_for(utype, src):
                raise ParseError(f"{_abbr(src)} cannot support into "
                                 f"{_abbr(dst)} (not adjacent)")
            return src, {"type": "SUPPORT", "attacker": atk, "target": dst}
        sh = re.match(r"S\s+[AF]?\s*([A-Za-z]{3})\s*$", rest, re.I)
        if sh:
            tgt = self._pid_from_token(sh.group(1))
            if tgt not in _adj_for(utype, src):
                raise ParseError(f"{_abbr(src)} cannot support hold of "
                                 f"{_abbr(tgt)} (not adjacent)")
            return src, {"type": "SUPPORT", "target": tgt}

        # MOVE: A PAR - BUR  (optionally VIA CONVOY)
        mm = re.match(r"-\s*([A-Za-z]{3})(.*)$", rest, re.I)
        if mm:
            dst = self._pid_from_token(mm.group(1))
            via = "convoy" in mm.group(2).lower()
            if via:
                if utype != "A":
                    raise ParseError("only armies are convoyed")
                # Reject multi-fleet convoy chains: caller can only name one
                # fleet via the matching C order; chains are not representable
                # in this syntax and we additionally guard at adjudication.
                return src, {"type": "MOVE", "target": dst, "via_convoy": True}
            if dst not in _adj_for(utype, src):
                raise ParseError(f"{_abbr(src)} cannot move to "
                                 f"{_abbr(dst)} (not adjacent for a {utype})")
            if not _can_occupy(utype, dst):
                raise ParseError(f"a {utype} cannot occupy {_abbr(dst)}")
            return src, {"type": "MOVE", "target": dst, "via_convoy": False}

        if up == "HOLD" or up == "H":
            return src, {"type": "HOLD"}

        raise ParseError(f"unrecognised order: {line!r}")

    def _parse_retreats(self, state: DipState, player: int, text: str) -> Action:
        power = POWERS[player]
        dis = set(self._dislodged_of(state, power))
        orders: Dict[int, dict] = {}
        m = self._RET_BLOCK_RE.search(text)
        if m:
            for raw in m.group(1).splitlines():
                line = raw.strip()
                if not line:
                    continue
                rm = re.match(r"[AF]?\s*([A-Za-z]{3})\s*-\s*([A-Za-z]{3})", line, re.I)
                if rm:
                    src = self._pid_from_token(rm.group(1))
                    dst = self._pid_from_token(rm.group(2))
                    if src not in dis:
                        raise ParseError(f"{_abbr(src)} is not a dislodged unit")
                    legal = self._legal_retreat_targets(state, src)
                    if dst not in legal:
                        raise ParseError(f"{_abbr(src)} cannot retreat to {_abbr(dst)}")
                    orders[src] = {"type": "RETREAT", "target": dst}
                    continue
                dm = re.match(r"[AF]?\s*([A-Za-z]{3})\s+DISBAND", line, re.I)
                if dm:
                    src = self._pid_from_token(dm.group(1))
                    if src not in dis:
                        raise ParseError(f"{_abbr(src)} is not a dislodged unit")
                    orders[src] = {"type": "DISBAND"}
                    continue
                raise ParseError(f"cannot parse retreat line: {line!r}")
        # Auto-resolve any unspecified dislodged units deterministically.
        for pid in dis:
            if pid not in orders:
                tgt = self._auto_retreat_target(state, pid)
                orders[pid] = ({"type": "RETREAT", "target": tgt} if tgt is not None
                               else {"type": "DISBAND"})
        return {"type": "submit_retreats", "orders": orders}

    def _legal_retreat_targets(self, state: DipState, pid: int) -> List[int]:
        utype = state.dislodged[pid]["type"]
        forbidden = state.dislodged[pid]["from"]
        out = []
        for nb in sorted(_adj_for(utype, pid)):
            if not _can_occupy(utype, nb):
                continue
            if nb in state.units or nb == forbidden or nb in state.contested:
                continue
            out.append(nb)
        return out

    def _parse_builds(self, state: DipState, player: int, text: str) -> Action:
        power = POWERS[player]
        delta = self._build_delta(state, power)
        orders: Dict[int, dict] = {}
        m = self._BUILD_BLOCK_RE.search(text)
        if m:
            for raw in m.group(1).splitlines():
                line = raw.strip()
                if not line:
                    continue
                bm = re.match(r"BUILD\s+([AF])\s+([A-Za-z]{3})", line, re.I)
                if bm:
                    utype = bm.group(1).upper()
                    pid = self._pid_from_token(bm.group(2))
                    if delta <= 0:
                        raise ParseError("you have no builds available")
                    if pid not in self._buildable_sites(state, power):
                        raise ParseError(f"{_abbr(pid)} is not a buildable home centre")
                    if not _can_occupy(utype, pid):
                        raise ParseError(f"cannot build a {utype} on {_abbr(pid)}")
                    orders[pid] = {"type": "BUILD", "unit": utype}
                    continue
                dm = re.match(r"DISBAND\s+[AF]?\s*([A-Za-z]{3})", line, re.I)
                if dm:
                    pid = self._pid_from_token(dm.group(1))
                    if delta >= 0:
                        raise ParseError("you have no disbands required")
                    if state.units.get(pid, (None, None))[1] != power:
                        raise ParseError(f"you do not own a unit on {_abbr(pid)}")
                    orders[pid] = {"type": "DISBAND"}
                    continue
                raise ParseError(f"cannot parse build line: {line!r}")
        return {"type": "submit_builds", "orders": orders}

    # -------------------------------------------------------------- step
    def step(self, state: DipState, action: Action) -> DipState:
        t = action.get("type")
        state.turn += 1

        # Negotiation / alliance actions delegate to the messaging mixin.
        if t in ("say", "whisper", "pass_talk", "alliance_propose",
                 "alliance_accept", "alliance_decline", "alliance_break"):
            return self.nego_step(state, action)

        if t == "advance_phase":
            return self._advance_phase(state)

        if t == "submit_orders" and state.phase in (PH_SPRING, PH_FALL):
            seat = self.active_player(state)
            if seat < 0:
                return state
            power = POWERS[seat]
            orders = action.get("orders", {})
            # Only accept orders for this power's units.
            for src, od in orders.items():
                if state.units.get(src, (None, None))[1] == power:
                    state.pending_orders[int(src)] = od
            state.last_orders_by_power[power] = {
                int(s): o for s, o in orders.items()
                if state.units.get(int(s), (None, None))[1] == power
            }
            state.orders_done[seat] = True
            if self.active_player(state) < 0:
                return self._resolve_movement(state)
            return state

        if t == "submit_retreats" and state.phase == PH_RETREAT:
            seat = self.active_player(state)
            if seat < 0:
                return state
            self._apply_retreats(state, seat, action.get("orders", {}))
            state.retreats_done[seat] = True
            if self.active_player(state) < 0:
                return self._finish_retreats(state)
            return state

        if t == "submit_builds" and state.phase == PH_WINTER:
            seat = self.active_player(state)
            if seat < 0:
                return state
            self._apply_builds(state, seat, action.get("orders", {}))
            state.builds_done[seat] = True
            if self.active_player(state) < 0:
                return self._finish_winter(state)
            return state

        return state

    def _advance_phase(self, state: DipState) -> DipState:
        """Called by the runner when active_player() == -1 mid-phase.

        For movement/retreat/winter this means the phase is complete and we
        resolve it; for negotiation the mixin already exits to return_phase, so
        a stray advance is a no-op.
        """
        if state.phase in (PH_SPRING, PH_FALL):
            return self._resolve_movement(state)
        if state.phase == PH_RETREAT:
            return self._finish_retreats(state)
        if state.phase == PH_WINTER:
            return self._finish_winter(state)
        return state

    # --------------------------------------------------- movement resolution
    def _resolve_movement(self, state: DipState) -> DipState:
        board = Board(units=dict(state.units))
        orders = dict(state.pending_orders)
        res = adjudicate(orders, board)

        # Honour/betray judgement on the just-submitted orders.
        self._judge_movement(state, orders, res)

        # B5: stash the SINGLE public resolution payload (every power's orders
        # + the resolved outcomes). ``observations`` emits this to all living
        # powers exactly once, only AFTER the whole movement phase resolves.
        state.last_resolution = {
            "type": "resolution",
            "phase": state.phase,
            "season": state.season,
            "year": state.year,
            "orders": {str(s): o for s, o in sorted(orders.items())},
            "moves": {str(s): d for s, d in sorted(res.moves.items())},
            "dislodged": {str(p): {"power": d["power"], "type": d["type"],
                                   "from": d["from"]}
                          for p, d in sorted(res.dislodged.items())},
            "bounces": sorted(res.bounces),
            "cut_supports": sorted(res.cut_supports),
            "failed_convoys": sorted(res.failed_convoys),
        }

        # Apply moves.
        new_units: Dict[int, Tuple[str, str]] = {}
        # Units that did not move stay (unless dislodged).
        for pid, unit in state.units.items():
            if pid in res.moves:
                continue
            if pid in res.dislodged:
                continue
            new_units[pid] = unit
        for src, dst in res.moves.items():
            new_units[dst] = state.units[src]

        state.units = new_units
        state.dislodged = dict(res.dislodged)
        state.contested = set(res.bounces)
        state.pending_orders = {}
        state.orders_done = [False] * self.n_players

        moved_desc = ", ".join(f"{_abbr(s)}->{_abbr(d)}" for s, d in
                               sorted(res.moves.items())) or "(no moves)"
        state.history.append(f"{state.season} {state.year} resolved: {moved_desc}")

        # Transition: retreat phase if anyone is dislodged.
        if state.dislodged:
            state.retreats_done = [False] * self.n_players
            state.phase = PH_RETREAT
            return state
        return self._after_movement(state)

    def _apply_retreats(self, state: DipState, seat: int, orders: Dict[int, dict]):
        power = POWERS[seat]
        for pid, od in orders.items():
            pid = int(pid)
            if pid not in state.dislodged or state.dislodged[pid]["power"] != power:
                continue
            if od.get("type") == "RETREAT":
                tgt = int(od["target"])
                if tgt in self._legal_retreat_targets(state, pid) \
                        and tgt not in state.units:
                    state.units[tgt] = (state.dislodged[pid]["type"], power)
                    state.history.append(f"{power} retreats to {_abbr(tgt)}")
            # DISBAND or illegal retreat -> unit is removed (already not in units).
            del state.dislodged[pid]

    def _finish_retreats(self, state: DipState) -> DipState:
        # Auto-resolve any remaining dislodged units (deterministic).
        for pid in list(state.dislodged.keys()):
            d = state.dislodged[pid]
            tgt = self._auto_retreat_target(state, pid)
            if tgt is not None and tgt not in state.units:
                state.units[tgt] = (d["type"], d["power"])
            del state.dislodged[pid]
        state.contested = set()
        return self._after_movement(state)

    def _after_movement(self, state: DipState) -> DipState:
        """Advance season after a movement (+ retreat) phase fully resolves."""
        if state.season == "Spring":
            # Move to Fall: negotiation then Fall movement.
            state.season = "Fall"
            if self.config.messaging:
                self.start_negotiation(state, return_phase=PH_FALL,
                                       rounds=self.config.nego_rounds)
            else:
                state.phase = PH_FALL
            return state
        # Fall complete: update SC ownership, then Winter / next year.
        self._update_supply_centres(state)
        if self._check_winner(state):
            return state
        # Winter adjustments only if any power has a non-zero build delta.
        if any(self._build_delta(state, POWERS[s]) != 0
               for s in range(self.n_players)):
            state.builds_done = [False] * self.n_players
            state.phase = PH_WINTER
            return state
        return self._advance_year(state)

    def _update_supply_centres(self, state: DipState) -> None:
        for pid, (utype, power) in state.units.items():
            if pid in SUPPLY_CENTRE:
                state.sc_owner[pid] = power

    def _finish_winter(self, state: DipState) -> DipState:
        return self._advance_year(state)

    def _advance_year(self, state: DipState) -> DipState:
        state.year += 1
        state.season = "Spring"
        if state.year > 1900 + MAX_YEARS:
            return self._cap_winner(state)
        if self.config.messaging:
            self.start_negotiation(state, return_phase=PH_SPRING,
                                   rounds=self.config.nego_rounds)
        else:
            state.phase = PH_SPRING
        return state

    def _apply_builds(self, state: DipState, seat: int, orders: Dict[int, dict]):
        power = POWERS[seat]
        delta = self._build_delta(state, power)
        if delta > 0:
            built = 0
            for pid, od in sorted(orders.items()):
                pid = int(pid)
                if built >= delta:
                    break
                if od.get("type") != "BUILD":
                    continue
                if pid in self._buildable_sites(state, power):
                    utype = od.get("unit", "A")
                    if not _can_occupy(utype, pid):
                        utype = "A"
                    state.units[pid] = (utype, power)
                    built += 1
            # Excess builds waived (documented simplification).
        elif delta < 0:
            need = -delta
            removed = 0
            specified = [int(p) for p, od in orders.items()
                         if od.get("type") == "DISBAND"
                         and state.units.get(int(p), (None, None))[1] == power]
            for pid in specified:
                if removed >= need:
                    break
                if pid in state.units:
                    del state.units[pid]
                    removed += 1
            # Auto-disband the remainder deterministically.
            if removed < need:
                for pid in self._auto_disband_order(state, power):
                    if removed >= need:
                        break
                    if pid in state.units:
                        del state.units[pid]
                        removed += 1

    # --------------------------------------------------- alliance judgement
    def judge_alliance(self, state: DipState, action: Action, actor: int):
        """Required override hook. Movement judgement is done in bulk in
        ``_judge_movement`` (it needs the full order set), so this per-action
        hook is intentionally a no-op."""
        return []

    def _judge_movement(self, state: DipState, orders: Dict[int, dict],
                        res: Resolution) -> None:
        """Emit honored/betrayed alliance_event records for the resolved phase.

        honored : a promised SUPPORT for an active ally's unit/move appears.
        betrayed: a MOVE into an active ally's occupied province or owned SC,
                  OR omitting a promised support (a support_pact with no support
                  order present is a betrayal).
        """
        turn = state.turn
        active = [al for al in state.alli.alliances.values() if al.status == "active"]
        if not active:
            return

        for al in active:
            for actor_seat in al.members:
                power = POWERS[actor_seat]
                ally_seats = [m for m in al.members if m != actor_seat]
                my_orders = {s: o for s, o in orders.items()
                             if state.units.get(s, (None, None))[1] == power}

                # --- Betrayal by hostile MOVE into an ally's province/SC ---
                betrayed = False
                victims: List[int] = []
                for src, od in my_orders.items():
                    if od.get("type") != "MOVE":
                        continue
                    dst = od["target"]
                    for ally in ally_seats:
                        ally_power = POWERS[ally]
                        ally_here = state.units.get(dst, (None, None))[1] == ally_power
                        ally_sc = state.sc_owner.get(dst) == ally_power
                        if ally_here or ally_sc:
                            betrayed = True
                            if ally not in victims:
                                victims.append(ally)
                if betrayed:
                    self._emit_alliance_event(
                        state, "betrayed", al, turn=turn, actor=actor_seat,
                        counterparty=victims,
                        action_ref={"reason": "move_into_ally"})
                    continue

                # --- Honour: a support order benefiting an ally appears ---
                honored = False
                for src, od in my_orders.items():
                    if od.get("type") != "SUPPORT":
                        continue
                    # support-hold of an ally's unit, or support-move whose
                    # supported attacker is an ally's unit.
                    if "attacker" in od:
                        atk = od["attacker"]
                        if state.units.get(atk, (None, None))[1] in \
                                [POWERS[a] for a in ally_seats]:
                            honored = True
                    else:
                        tgt = od["target"]
                        if state.units.get(tgt, (None, None))[1] in \
                                [POWERS[a] for a in ally_seats]:
                            honored = True
                if honored:
                    self._emit_alliance_event(
                        state, "honored", al, turn=turn, actor=actor_seat,
                        counterparty=ally_seats,
                        action_ref={"reason": "supported_ally"})
                elif al.kind == "support_pact":
                    # A support_pact with no supporting order present is an
                    # omission betrayal (only judged when the pact promised it).
                    self._emit_alliance_event(
                        state, "betrayed", al, turn=turn, actor=actor_seat,
                        counterparty=ally_seats,
                        action_ref={"reason": "omitted_promised_support"})

    def _emit_alliance_event(self, state, event, al, *, turn, actor,
                             counterparty, action_ref):
        ev = new_event(event, al, turn=turn, actor=actor,
                       counterparty=counterparty, action_ref=action_ref)
        state.alli.events.append(ev)
        state.history.append(
            f"{POWERS[actor]} {event} alliance #{al.id} ({al.kind})")

    # ----------------------------------------------------- win checking
    def _check_winner(self, state: DipState) -> bool:
        for s in range(self.n_players):
            if self._sc_count(state, POWERS[s]) >= SOLO_SC:
                state.winner = s
                state.win_reason = f"{POWERS[s]} solo victory ({SOLO_SC}+ SCs)"
                state.phase = PH_TERMINAL
                return True
        alive = self.living_seats(state)
        if len(alive) == 1:
            state.winner = alive[0]
            state.win_reason = f"{POWERS[alive[0]]} last power standing"
            state.phase = PH_TERMINAL
            return True
        return False

    def _cap_winner(self, state: DipState) -> DipState:
        counts = {s: self._sc_count(state, POWERS[s]) for s in range(self.n_players)}
        best = max(counts.values()) if counts else 0
        winner = min(s for s in counts if counts[s] == best)
        state.winner = winner
        state.win_reason = (f"year cap ({MAX_YEARS}y): {POWERS[winner]} leads "
                            f"with {best} SCs")
        state.phase = PH_TERMINAL
        return state

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: DipState) -> bool:
        return state.phase == PH_TERMINAL

    def rewards(self, state: DipState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        return [1.0 if state.winner == p else 0.0 for p in range(self.n_players)]

    # ------------------------------------------------------- observations
    def observations(self, prev_state, new_state, action: Action,
                     actor: int) -> List[Obs]:
        t = action.get("type")
        if t in ("say", "whisper", "pass_talk", "alliance_propose",
                 "alliance_accept", "alliance_decline", "alliance_break"):
            return self.nego_observations(prev_state, new_state, action, actor)

        # B5 (hidden-info leak fix): during a movement phase, orders are
        # submitted SEQUENTIALLY by seat. A power's submitted orders must NOT be
        # routed to powers that have not yet submitted — that would leak their
        # plan. We therefore:
        #   * route a submit_orders Obs ONLY to the actor (audience=[actor]),
        #     keeping the full orders in the god-LOG (log=full content); and
        #   * emit ONE public 'resolution' Obs (audience=all living powers,
        #     carrying EVERY power's orders + outcomes) only AFTER the whole
        #     movement phase has resolved.
        # The movement phase has resolved exactly when ``_resolve_movement`` has
        # set ``new_state.last_resolution`` (it is cleared here once consumed).
        if t == "submit_orders":
            obs_list: List[Obs] = []
            full_log = {
                "type": "submit_orders",
                "actor": actor,
                "season": new_state.season,
                "year": new_state.year,
                "orders": {str(s): o for s, o in
                           action.get("orders", {}).items()},
            }
            # Private acknowledgement to the submitting power only. The actor
            # already knows its own orders; bystanders learn nothing. The full
            # content lives in the god-LOG.
            obs_list.append(Obs(audience=[actor],
                                payload={"type": "orders_ack", "actor": actor,
                                         "season": new_state.season,
                                         "year": new_state.year},
                                log=full_log))
            res = getattr(new_state, "last_resolution", None)
            if res is not None:
                living = list(self.living_seats(new_state))
                obs_list.append(Obs(audience=living, payload=dict(res)))
                new_state.last_resolution = None
            return obs_list

        if t == "advance_phase":
            # A movement phase may resolve via advance_phase (when no power had
            # any units to order). Emit the pending public resolution, if any.
            res = getattr(new_state, "last_resolution", None)
            if res is not None:
                living = list(self.living_seats(new_state))
                new_state.last_resolution = None
                return [Obs(audience=living, payload=dict(res))]
            return []

        if t in ("submit_retreats", "submit_builds"):
            living = list(self.living_seats(new_state))
            payload = {"type": "resolution", "kind": t, "actor": actor,
                       "orders": {str(s): o for s, o in
                                  action.get("orders", {}).items()}}
            return [Obs(audience=living, payload=payload)]

        # Fallback: broadcast.
        return [Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

    # ------------------------------------------------------- watcher hooks
    def god_view(self, state: DipState) -> dict:
        return {
            "powers": list(POWERS),
            "units": {_abbr(p): {"type": t, "power": o}
                      for p, (t, o) in sorted(state.units.items())},
            "sc_owner": {_abbr(p): o for p, o in sorted(state.sc_owner.items())},
            "adjudicator": self.adjudicator,
        }

    def snapshot(self, state: DipState) -> dict:
        public = {
            "season": state.season, "year": state.year, "phase": state.phase,
            "units": {_abbr(p): {"type": t, "power": o}
                      for p, (t, o) in sorted(state.units.items())},
            "sc_owner": {_abbr(p): o for p, o in sorted(state.sc_owner.items())},
            "sc_counts": {POWERS[s]: self._sc_count(state, POWERS[s])
                          for s in range(self.n_players)},
        }
        hidden = {
            "dislodged": {_abbr(p): d for p, d in state.dislodged.items()},
            "pending_orders": {_abbr(int(p)): o
                               for p, o in state.pending_orders.items()},
        }
        return {"public": public, "hidden": hidden}

    def render_board(self, state: DipState, *, reveal: str = "god") -> str:
        lines = [f"=== Diplomacy {state.season} {state.year} ({state.phase}) ==="]
        sc_line = "  ".join(
            f"{POWERS[s][:3]}:{self._sc_count(state, POWERS[s])}SC/"
            f"{len(self._units_of(state, POWERS[s]))}u"
            for s in range(self.n_players)
        )
        lines.append(sc_line)
        # Units grouped by power.
        for s in range(self.n_players):
            power = POWERS[s]
            us = sorted(self._units_of(state, power))
            if not us:
                continue
            glyphs = " ".join(f"{state.units[u][0]}{_abbr(u)}" for u in us)
            lines.append(f"  {power[:3]}: {glyphs}")
        if state.dislodged:
            dl = " ".join(f"{d['type']}{_abbr(p)}({d['power'][:3]})"
                          for p, d in sorted(state.dislodged.items()))
            lines.append(f"  dislodged: {dl}")
        return "\n".join(lines)
