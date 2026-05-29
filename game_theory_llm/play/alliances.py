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
                    raise ParseError("cannot propose an alliance to yourself")
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
def alliance_summary(alli: "Optional[AllianceState]") -> dict:
    """Reduce an AllianceState's event trail to the terminal-record summary
    (spec §4). Pure function over ``events`` + ``alliances`` — safe to call
    on a fresh or empty state.
    """
    empty = {
        "n_proposed": 0, "n_accepted": 0, "n_declined": 0, "n_broken": 0,
        "n_honored": 0, "n_betrayed": 0, "per_player": {},
    }
    if alli is None:
        return empty

    per: Dict[int, Dict[str, int]] = {}

    def _slot(seat: int) -> Dict[str, int]:
        return per.setdefault(int(seat), {
            "proposed": 0, "accepted": 0, "honored": 0,
            "betrayed": 0, "betrayed_against": 0,
        })

    counts = {"propose": 0, "accept": 0, "decline": 0, "break": 0,
              "honored": 0, "betrayed": 0, "expired": 0}
    for ev in alli.events:
        e = ev.get("event")
        if e in counts:
            counts[e] += 1
        actor = ev.get("actor")
        cp = ev.get("counterparty", []) or []
        if e == "propose" and actor is not None:
            _slot(actor)["proposed"] += 1
        elif e == "accept" and actor is not None:
            _slot(actor)["accepted"] += 1
        elif e == "honored" and actor is not None:
            _slot(actor)["honored"] += 1
        elif e == "betrayed" and actor is not None:
            _slot(actor)["betrayed"] += 1
            for victim in cp:
                _slot(victim)["betrayed_against"] += 1

    return {
        "n_proposed": counts["propose"],
        "n_accepted": counts["accept"],
        "n_declined": counts["decline"],
        "n_broken": counts["break"],
        "n_honored": counts["honored"],
        "n_betrayed": counts["betrayed"],
        "per_player": {str(k): v for k, v in sorted(per.items())},
    }
