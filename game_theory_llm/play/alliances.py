"""Shared alliance machinery (propose / accept / decline / break + honour /
betray instrumentation).

Alliances are *cheap talk by default* — betrayal must remain physically
possible so the experiment can measure it. The only hard-enforcement knob is
``GameConfig.enforce_alliances`` (Risk), and that lives in the game, not here.

The data model (``Alliance`` / ``AllianceState``) plus the ``AllianceMixin``
behaviour are game-agnostic. Per-game hooks (``alliance_effects``,
``judge_alliance``, ``alliance_legal_actions``) carry sensible defaults that a
game overrides to wire alliances into its own board-resolution point.

See spec §3.3 / §4 / §8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import GOD, Action, Obs, ParseError


# ---------------------------------------------------------------- data model
@dataclass
class Alliance:
    """One alliance ledger entry.

    ``status`` transitions: ``proposed`` -> (``active`` | ``declined``);
    ``active`` -> (``broken`` | ``expired`` | ``honored``). Honour/betray are
    *judged* by the game at its board-resolution point and recorded as
    ``alliance_event`` records; they do not necessarily flip ``status``
    (an active alliance can be honoured repeatedly).
    """

    id: int
    members: List[int]            # sorted seats (>=2): proposer + invitees
    proposer: int
    kind: str                     # per-game enum (see spec §5)
    terms: dict                   # freeform, game-specific
    status: str                   # proposed|active|declined|broken|expired|honored
    proposed_turn: int
    accepted_turn: Optional[int] = None
    broken_turn: Optional[int] = None
    broken_by: Optional[int] = None
    pending: List[int] = field(default_factory=list)   # invitees yet to accept


@dataclass
class AllianceState:
    next_id: int = 0
    alliances: Dict[int, Alliance] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)   # audit trail (mirrors JSONL)


# ----------------------------------------------------------------- event util
def new_event(event: str, alliance: Alliance, *, turn: int, actor: int,
              counterparty: "Optional[List[int]]" = None,
              action_ref: "Optional[dict]" = None) -> dict:
    """Build a canonical ``alliance_event`` dict (spec §4).

    ``alliance_age_turns`` is computed from ``accepted_turn`` when the
    alliance is active, else None.
    """
    age: Optional[int] = None
    if alliance.accepted_turn is not None:
        age = max(0, turn - alliance.accepted_turn)
    return {
        "type": "alliance_event",
        "turn": turn,
        "event": event,
        "alliance_id": alliance.id,
        "kind": alliance.kind,
        "proposer": alliance.proposer,
        "members": list(alliance.members),
        "actor": actor,
        "counterparty": list(counterparty or []),
        "terms": dict(alliance.terms),
        "action_ref": action_ref,
        "alliance_age_turns": age,
    }


# ----------------------------------------------------------------- parse tags
# <ally propose to=2,3 kind=nonaggression>terms text</ally>
_PROPOSE_RE = re.compile(
    r"<ally\s+propose\b([^>]*)>(.*?)</ally>",
    re.IGNORECASE | re.DOTALL,
)
_TO_RE = re.compile(r"\bto\s*=\s*([0-9,\s]+)", re.IGNORECASE)
_KIND_RE = re.compile(r"\bkind\s*=\s*([A-Za-z_][A-Za-z0-9_\-]*)", re.IGNORECASE)
# <ally accept 7>  <ally decline 7>
_ACCEPT_RE = re.compile(r"<ally\s+accept\s+(\d+)\s*/?>", re.IGNORECASE)
_DECLINE_RE = re.compile(r"<ally\s+decline\s+(\d+)\s*/?>", re.IGNORECASE)
# <ally break 7>reason</ally>  (reason optional)
_BREAK_RE = re.compile(
    r"<ally\s+break\s+(\d+)\s*>(.*?)</ally>",
    re.IGNORECASE | re.DOTALL,
)
_BREAK_SELF_RE = re.compile(r"<ally\s+break\s+(\d+)\s*/?>", re.IGNORECASE)


def _parse_seat_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.replace(" ", "").split(","):
        if tok == "":
            continue
        out.append(int(tok))
    return out


# ------------------------------------------------------------------- mixin
class AllianceMixin:
    """Reusable alliance behaviour. A game composes it as a base class and
    keeps an ``AllianceState`` on its state object under attribute ``alli``.
    """

    # ---- membership / queries ----
    def is_allied(self, state, a: int, b: int) -> bool:
        """True iff seats ``a`` and ``b`` share an ``active`` alliance."""
        alli = getattr(state, "alli", None)
        if alli is None:
            return False
        for al in alli.alliances.values():
            if al.status == "active" and a in al.members and b in al.members:
                return True
        return False

    # ---- legal-action enumeration (default; games may extend) ----
    def alliance_legal_actions(self, state, player: int) -> List[Action]:
        """Default enumeration: accept/decline any alliance pending this
        player, break any active alliance this player is in, and a single
        generic nonaggression proposal to each other living seat.

        Games with richer ``kind`` enums override this; the defaults keep
        ``RandomPlayer`` smoke-tests exercising the full lifecycle.
        """
        alli = getattr(state, "alli", None)
        if alli is None:
            return []
        actions: List[Action] = []
        living = list(self.living_seats(state)) if hasattr(self, "living_seats") \
            else list(range(getattr(self, "n_players", 0)))
        for al in alli.alliances.values():
            if al.status == "proposed" and player in al.pending:
                actions.append({"type": "alliance_accept", "alliance_id": al.id})
                actions.append({"type": "alliance_decline", "alliance_id": al.id})
            if al.status == "active" and player in al.members:
                actions.append({"type": "alliance_break", "alliance_id": al.id,
                                "reason": ""})
        for other in living:
            if other != player:
                actions.append({"type": "alliance_propose", "to": [other],
                                "kind": "nonaggression", "terms": {}})
        return actions

    # ---- parsing ----
    def alliance_parse(self, state, player: int, text: str) -> Action:
        """Parse an ``<ally ...>`` tag into an alliance Action.

        Raises ParseError if no alliance tag is present (callers fall through
        to message parsing). Structure only — never quality.
        """
        m = _PROPOSE_RE.search(text)
        if m:
            attrs, terms_text = m.group(1), m.group(2)
            to_m = _TO_RE.search(attrs)
            if not to_m:
                raise ParseError("ally propose requires to=<seat,...>")
            to = _parse_seat_list(to_m.group(1))
            kind_m = _KIND_RE.search(attrs)
            kind = kind_m.group(1).lower() if kind_m else "nonaggression"
            living = set(self.living_seats(state)) if hasattr(self, "living_seats") \
                else set(range(getattr(self, "n_players", 0)))
            cleaned: List[int] = []
            for seat in to:
                if seat == player:
                    # Same slip as self-whispers: models list every seat in the
                    # group including their own. Drop the proposer; a proposal
                    # addressed only to self still fails the emptiness check.
                    continue
                if seat not in living:
                    raise ParseError(f"seat {seat} is not a living target")
                if seat not in cleaned:
                    cleaned.append(seat)
            if not cleaned:
                raise ParseError("ally propose requires at least one invitee")
            return {"type": "alliance_propose", "to": sorted(cleaned),
                    "kind": kind, "terms": {"text": terms_text.strip()}}
        m = _ACCEPT_RE.search(text)
        if m:
            return {"type": "alliance_accept", "alliance_id": int(m.group(1))}
        m = _DECLINE_RE.search(text)
        if m:
            return {"type": "alliance_decline", "alliance_id": int(m.group(1))}
        m = _BREAK_RE.search(text)
        if m:
            return {"type": "alliance_break", "alliance_id": int(m.group(1)),
                    "reason": m.group(2).strip()}
        m = _BREAK_SELF_RE.search(text)
        if m:
            return {"type": "alliance_break", "alliance_id": int(m.group(1)),
                    "reason": ""}
        raise ParseError("no <ally ...> tag found")

    # ---- application (mutation) ----
    def apply_alliance_action(self, state, actor: int,
                              action: Action) -> List[dict]:
        """Apply an alliance Action to ``state.alli`` and return the list of
        ``alliance_event`` records produced. Mutates the AllianceState.
        """
        alli: AllianceState = getattr(state, "alli")
        turn = getattr(state, "turn", 0)
        t = action.get("type")
        events: List[dict] = []

        if t == "alliance_propose":
            to = sorted({int(x) for x in action.get("to", []) if int(x) != actor})
            members = sorted(set(to) | {actor})
            al = Alliance(
                id=alli.next_id,
                members=members,
                proposer=actor,
                kind=str(action.get("kind", "nonaggression")),
                terms=dict(action.get("terms", {})),
                status="proposed",
                proposed_turn=turn,
                pending=list(to),
            )
            alli.alliances[al.id] = al
            alli.next_id += 1
            ev = new_event("propose", al, turn=turn, actor=actor, counterparty=to)
            events.append(ev)

        elif t == "alliance_accept":
            aid = int(action.get("alliance_id", -1))
            al = alli.alliances.get(aid)
            if al is not None and al.status == "proposed" and actor in al.pending:
                al.pending = [s for s in al.pending if s != actor]
                ev = new_event("accept", al, turn=turn, actor=actor,
                               counterparty=[al.proposer])
                events.append(ev)
                if not al.pending:
                    al.status = "active"
                    al.accepted_turn = turn

        elif t == "alliance_decline":
            aid = int(action.get("alliance_id", -1))
            al = alli.alliances.get(aid)
            if al is not None and al.status == "proposed" and actor in al.pending:
                al.status = "declined"
                al.pending = [s for s in al.pending if s != actor]
                ev = new_event("decline", al, turn=turn, actor=actor,
                               counterparty=[al.proposer])
                events.append(ev)

        elif t == "alliance_break":
            aid = int(action.get("alliance_id", -1))
            al = alli.alliances.get(aid)
            if al is not None and al.status == "active" and actor in al.members:
                al.status = "broken"
                al.broken_turn = turn
                al.broken_by = actor
                if action.get("reason"):
                    al.terms = dict(al.terms)
                    al.terms["break_reason"] = action["reason"]
                cp = [s for s in al.members if s != actor]
                ev = new_event("break", al, turn=turn, actor=actor,
                               counterparty=cp)
                events.append(ev)

        alli.events.extend(events)
        return events

    # ---- routing / masking ----
    def alliance_observations(self, state, action: Action,
                              actor: int) -> List[Obs]:
        """Route the most recent alliance transition.

        propose/accept/decline -> members ∪ pending (+ full god-log);
        break -> all affected members AND a public broadcast (a betrayal is
        observable to its victims by definition).
        """
        alli: AllianceState = getattr(state, "alli")
        aid = action.get("alliance_id")
        if aid is None and action.get("type") == "alliance_propose":
            # The just-created alliance is the highest id.
            aid = alli.next_id - 1
        al = alli.alliances.get(int(aid)) if aid is not None else None
        if al is None:
            return []
        t = action.get("type")
        payload = {
            "type": "alliance_event",
            "event": t.replace("alliance_", "") if t else "",
            "alliance_id": al.id,
            "kind": al.kind,
            "proposer": al.proposer,
            "members": list(al.members),
            "actor": actor,
            "status": al.status,
        }
        if t == "alliance_break":
            living = list(self.living_seats(state)) if hasattr(self, "living_seats") \
                else list(range(getattr(self, "n_players", 0)))
            audience = sorted(set(living) | set(al.members))
            return [Obs(audience=audience, payload=payload)]
        audience = sorted(set(al.members) | set(al.pending))
        return [Obs(audience=audience, payload=payload)]

    # ---- per-game hooks (defaults) ----
    def alliance_effects(self, state) -> dict:
        """Per-game mechanical effects of active alliances. Default {}."""
        return {}

    def judge_alliance(self, state, action: Action, actor: int) -> List[dict]:
        """Per-game: emit honored/betrayed events at board resolution.
        Default: no judgement."""
        return []


# ------------------------------------------------------------- summary reduction
def _empty_summary() -> dict:
    return {
        "n_proposed": 0, "n_accepted": 0, "n_declined": 0, "n_broken": 0,
        "n_honored": 0, "n_betrayed": 0,
        "honored_events": 0, "betrayed_events": 0, "expired_events": 0,
        "per_player": {},
    }


def summary_from_alliance_events(events: "List[dict]") -> dict:
    """Per-alliance, bounded reduction of a list of ``alliance_event`` records.

    This is the single source of truth shared by ``alliance_summary`` (live
    state) and ``metrics`` (log replay) so the live and replayed numbers agree
    (B6/M3/M4). Each *distinct* ``alliance_id`` is classified by lifecycle
    state and, if it ever became active, into exactly ONE terminal bucket:

      * ``betrayed`` if ANY ``betrayed`` event references its alliance_id
        (betrayal dominates),
      * else ``honored`` (an active alliance with no betrayal counts as
        honoured — this gives Risk, which only emits ``betrayed`` events, a
        meaningful honour rate; m1).

    Counts are therefore per-alliance and bounded:
      ``n_honored + n_betrayed == n_accepted`` and every rate lands in [0, 1].
    Raw per-occasion judgement tallies are preserved as the diagnostic fields
    ``honored_events`` / ``betrayed_events`` / ``expired_events``.

    ``events`` may be either the canonical ``alliance_event`` records (which
    carry ``event`` + ``alliance_id``) or the audit-trail dicts on
    ``AllianceState.events`` (same shape). Records without an ``alliance_id``
    are ignored for the per-alliance buckets.
    """
    # Per-alliance lifecycle bookkeeping.
    proposed_ids: set = set()
    declined_ids: set = set()
    broken_ids: set = set()
    active_ids: set = set()             # ever reached "accepted"/active
    betrayed_ids: set = set()           # had >=1 betrayed event
    honored_event_ids: set = set()      # had >=1 honored event (diagnostic)

    # Raw per-occasion diagnostic tallies.
    honored_events = betrayed_events = expired_events = 0

    per: Dict[int, Dict[str, int]] = {}

    def _slot(seat: int) -> Dict[str, int]:
        return per.setdefault(int(seat), {
            "proposed": 0, "accepted": 0, "honored": 0,
            "betrayed": 0, "betrayed_against": 0,
        })

    # First-pass per-alliance proposer/acceptor seats (dedupe per alliance).
    proposer_of: Dict[int, int] = {}
    acceptors_of: Dict[int, set] = {}
    members_of: Dict[int, set] = {}      # union of members seen for the alliance

    def _note_members(aid, ev):
        ms = ev.get("members")
        if ms:
            members_of.setdefault(aid, set()).update(int(m) for m in ms)

    for ev in events or []:
        e = ev.get("event")
        aid = ev.get("alliance_id")
        actor = ev.get("actor")
        cp = ev.get("counterparty", []) or []
        if aid is not None:
            _note_members(aid, ev)
        if e == "propose":
            if aid is not None:
                proposed_ids.add(aid)
                if actor is not None:
                    proposer_of[aid] = int(actor)
        elif e == "accept":
            if aid is not None:
                proposed_ids.add(aid)      # an accept implies a proposal
                acceptors_of.setdefault(aid, set())
                if actor is not None:
                    acceptors_of[aid].add(int(actor))
        elif e == "decline":
            if aid is not None:
                proposed_ids.add(aid)
                declined_ids.add(aid)
        elif e == "break":
            if aid is not None:
                proposed_ids.add(aid)
                active_ids.add(aid)        # only active alliances can break
                broken_ids.add(aid)
        elif e == "honored":
            honored_events += 1
            if aid is not None:
                proposed_ids.add(aid)
                active_ids.add(aid)
                honored_event_ids.add(aid)
            if actor is not None:
                _slot(actor)["honored"] += 1
        elif e == "betrayed":
            betrayed_events += 1
            if aid is not None:
                proposed_ids.add(aid)
                active_ids.add(aid)
                betrayed_ids.add(aid)
            if actor is not None:
                _slot(actor)["betrayed"] += 1
                for victim in cp:
                    _slot(victim)["betrayed_against"] += 1
        elif e == "expired":
            expired_events += 1
            if aid is not None:
                proposed_ids.add(aid)
                active_ids.add(aid)

    # An alliance reaches "active" when ALL invitees (members minus proposer)
    # have accepted. We may not have seen a propose event for it (judgement-only
    # logs / fixtures), so infer membership from the union of members seen.
    for aid, acceptors in acceptors_of.items():
        if aid in declined_ids:
            continue
        members = members_of.get(aid, set())
        proposer = proposer_of.get(aid)
        invitees = {m for m in members if m != proposer} if members else set()
        if invitees and acceptors >= invitees:
            active_ids.add(aid)
        elif not members:
            # No membership info at all: treat any accept as activating
            # (best effort for sparse logs).
            active_ids.add(aid)

    # Per-alliance terminal classification: betrayal dominates.
    betrayed_alliances = {aid for aid in active_ids if aid in betrayed_ids}
    honored_alliances = active_ids - betrayed_alliances

    # Per-player proposed / accepted (deduped per alliance).
    for aid, seat in proposer_of.items():
        _slot(seat)["proposed"] += 1
    for aid, seats in acceptors_of.items():
        for seat in seats:
            _slot(seat)["accepted"] += 1

    return {
        "n_proposed": len(proposed_ids),
        "n_accepted": len(active_ids),
        "n_declined": len(declined_ids),
        "n_broken": len(broken_ids),
        "n_honored": len(honored_alliances),
        "n_betrayed": len(betrayed_alliances),
        "honored_events": honored_events,
        "betrayed_events": betrayed_events,
        "expired_events": expired_events,
        "per_player": {str(k): v for k, v in sorted(per.items())},
    }


def alliance_summary(alli: "Optional[AllianceState]") -> dict:
    """Reduce an AllianceState to the terminal-record summary (spec §4).

    Per-alliance and bounded: each alliance that ever became active is in
    exactly ONE of {honored, betrayed} (betrayal dominates), so the headline
    ``n_honored`` / ``n_betrayed`` rates are in [0, 1]. The raw per-occasion
    judgement counts survive as ``honored_events`` / ``betrayed_events`` /
    ``expired_events`` diagnostics. Pure function over the event trail — safe
    on a fresh or empty state.

    Because the audit trail does not always emit an explicit ``accept`` event
    for every alliance that reached ``active`` (a game may construct an
    alliance straight into ``active`` for tests, or only log judgement
    events), we reconcile the event-derived view with the live ledger so that
    every alliance whose ``accepted_turn`` is set is counted as accepted and
    classified.
    """
    if alli is None:
        return _empty_summary()

    # Synthesize a complete event view from the live ledger so an alliance that
    # is active in ``alli.alliances`` but lacks explicit propose/accept events
    # (e.g. test fixtures, or judgement-only games) is still counted.
    synth: List[dict] = []
    for al in alli.alliances.values():
        members = list(al.members)
        synth.append({"event": "propose", "alliance_id": al.id,
                      "actor": al.proposer, "members": members})
        accepted = (al.accepted_turn is not None or al.status in
                    ("active", "broken", "expired", "honored"))
        if accepted:
            for m in al.members:
                if m != al.proposer:
                    synth.append({"event": "accept", "alliance_id": al.id,
                                  "actor": m, "members": members})
        if al.status == "declined":
            synth.append({"event": "decline", "alliance_id": al.id,
                          "actor": al.proposer, "members": members})
        if al.status == "broken":
            synth.append({"event": "break", "alliance_id": al.id,
                          "actor": al.broken_by, "members": members})

    # The real judgement events (honored/betrayed/expired) plus any explicit
    # propose/accept/decline/break the audit trail recorded take precedence for
    # per-player attribution, so append them after the synthetic scaffold.
    return summary_from_alliance_events(synth + list(alli.events))
