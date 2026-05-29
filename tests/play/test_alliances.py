"""Alliance lifecycle + summary math.

Covers: propose -> accept -> active (only when ALL invitees accept);
decline by an invitee sets status=declined; break sets status=broken and
broken_by=actor; alliance_summary counts honored/betrayed correctly from the
event trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from game_theory_llm.play import (
    AllianceMixin,
    AllianceState,
    GameConfig,
    alliance_summary,
    new_event,
)
from game_theory_llm.play.base import Game


@dataclass
class _AlliState:
    phase: str = "play"
    alli: AllianceState = field(default_factory=AllianceState)
    turn: int = 0


class AllianceStub(AllianceMixin, Game):
    name = "alliance_stub"
    n_players = 4

    def initial_state(self, rng) -> _AlliState:
        return _AlliState()

    def living_seats(self, state) -> List[int]:
        return list(range(self.n_players))

    # abstract satisfiers
    def active_player(self, state) -> int:
        return -1

    def legal_actions(self, state, player: int):
        return self.alliance_legal_actions(state, player)

    def render_prompt(self, state, player: int) -> str:
        return ""

    def parse_action(self, state, player: int, text: str):
        return self.alliance_parse(state, player, text)

    def step(self, state, action):
        self.apply_alliance_action(state, action.get("actor", 0), action)
        return state

    def is_terminal(self, state) -> bool:
        return state.phase == "terminal"

    def rewards(self, state):
        return [0.0] * self.n_players


def test_propose_then_accept_becomes_active():
    game = AllianceStub()
    st = game.initial_state(None)
    # Seat 0 proposes a 3-way alliance with seats 1 and 2.
    evs = game.apply_alliance_action(
        st, 0, {"type": "alliance_propose", "to": [1, 2],
                "kind": "nonaggression", "terms": {}})
    assert len(evs) == 1 and evs[0]["event"] == "propose"
    al = st.alli.alliances[0]
    assert al.status == "proposed"
    assert sorted(al.members) == [0, 1, 2]
    assert sorted(al.pending) == [1, 2]

    # One acceptance is not enough — still proposed.
    game.apply_alliance_action(st, 1, {"type": "alliance_accept", "alliance_id": 0})
    assert al.status == "proposed"
    assert al.pending == [2]

    # Second acceptance -> active.
    game.apply_alliance_action(st, 2, {"type": "alliance_accept", "alliance_id": 0})
    assert al.status == "active"
    assert al.pending == []
    assert al.accepted_turn is not None
    assert game.is_allied(st, 0, 1) and game.is_allied(st, 1, 2)


def test_decline_sets_declined():
    game = AllianceStub()
    st = game.initial_state(None)
    game.apply_alliance_action(
        st, 0, {"type": "alliance_propose", "to": [1], "kind": "truce", "terms": {}})
    game.apply_alliance_action(st, 1, {"type": "alliance_decline", "alliance_id": 0})
    al = st.alli.alliances[0]
    assert al.status == "declined"
    assert not game.is_allied(st, 0, 1)


def test_break_sets_broken_by():
    game = AllianceStub()
    st = game.initial_state(None)
    game.apply_alliance_action(
        st, 0, {"type": "alliance_propose", "to": [1], "kind": "pact", "terms": {}})
    game.apply_alliance_action(st, 1, {"type": "alliance_accept", "alliance_id": 0})
    assert st.alli.alliances[0].status == "active"

    st.turn = 5
    evs = game.apply_alliance_action(
        st, 1, {"type": "alliance_break", "alliance_id": 0, "reason": "betrayal"})
    al = st.alli.alliances[0]
    assert al.status == "broken"
    assert al.broken_by == 1
    assert al.broken_turn == 5
    assert evs[0]["event"] == "break"
    assert evs[0]["counterparty"] == [0]
    assert al.terms.get("break_reason") == "betrayal"
    assert not game.is_allied(st, 0, 1)


def test_routing_propose_to_members_and_pending():
    game = AllianceStub()
    st = game.initial_state(None)
    action = {"type": "alliance_propose", "to": [1, 2],
              "kind": "nonaggression", "terms": {}}
    game.apply_alliance_action(st, 0, action)
    obs = game.alliance_observations(st, action, 0)
    assert len(obs) == 1
    assert sorted(obs[0].audience) == [0, 1, 2]
    assert obs[0].payload["event"] == "propose"


def test_routing_break_is_public():
    game = AllianceStub()
    st = game.initial_state(None)
    game.apply_alliance_action(
        st, 0, {"type": "alliance_propose", "to": [1], "kind": "pact", "terms": {}})
    game.apply_alliance_action(st, 1, {"type": "alliance_accept", "alliance_id": 0})
    action = {"type": "alliance_break", "alliance_id": 0, "reason": "x"}
    game.apply_alliance_action(st, 1, action)
    obs = game.alliance_observations(st, action, 1)
    # break is broadcast to all living seats.
    assert sorted(obs[0].audience) == [0, 1, 2, 3]
    assert obs[0].payload["event"] == "break"


def test_alliance_summary_counts():
    """Build a synthetic event trail and check the §4 reduction."""
    alli = AllianceState()
    # Two alliances proposed; one accepted (by two invitees), one declined.
    a0 = _make(alli, proposer=0, members=[0, 1], kind="pact")
    a0.status = "active"; a0.accepted_turn = 1
    alli.events.append(new_event("propose", a0, turn=0, actor=0, counterparty=[1]))
    alli.events.append(new_event("accept", a0, turn=1, actor=1, counterparty=[0]))

    a1 = _make(alli, proposer=2, members=[2, 3], kind="pact")
    alli.events.append(new_event("propose", a1, turn=0, actor=2, counterparty=[3]))
    a1.status = "declined"
    alli.events.append(new_event("decline", a1, turn=1, actor=3, counterparty=[2]))

    # Honour by seat 0 (toward 1); betrayal by seat 1 (against 0).
    alli.events.append(new_event("honored", a0, turn=2, actor=0, counterparty=[1]))
    alli.events.append(new_event("betrayed", a0, turn=3, actor=1, counterparty=[0]))
    a0.status = "broken"; a0.broken_by = 1
    alli.events.append(new_event("break", a0, turn=3, actor=1, counterparty=[0]))

    s = alliance_summary(alli)
    assert s["n_proposed"] == 2
    assert s["n_accepted"] == 1
    assert s["n_declined"] == 1
    assert s["n_broken"] == 1
    assert s["n_honored"] == 1
    assert s["n_betrayed"] == 1

    per = s["per_player"]
    assert per["0"]["proposed"] == 1
    assert per["0"]["honored"] == 1
    assert per["1"]["accepted"] == 1
    assert per["1"]["betrayed"] == 1
    assert per["0"]["betrayed_against"] == 1
    assert per["2"]["proposed"] == 1


def test_alliance_summary_empty_state():
    assert alliance_summary(None)["n_proposed"] == 0
    assert alliance_summary(AllianceState())["n_betrayed"] == 0


def _make(alli: AllianceState, *, proposer: int, members: List[int], kind: str):
    from game_theory_llm.play import Alliance

    al = Alliance(id=alli.next_id, members=sorted(members), proposer=proposer,
                  kind=kind, terms={}, status="proposed", proposed_turn=0,
                  pending=[m for m in members if m != proposer])
    alli.alliances[al.id] = al
    alli.next_id += 1
    return al


def test_parse_alliance_tags():
    game = AllianceStub()
    st = game.initial_state(None)
    a = game.alliance_parse(st, 0, "<ally propose to=1,2 kind=coalition>let's team</ally>")
    assert a["type"] == "alliance_propose"
    assert a["to"] == [1, 2]
    assert a["kind"] == "coalition"
    assert "team" in a["terms"]["text"]

    assert game.alliance_parse(st, 0, "<ally accept 7>")["alliance_id"] == 7
    assert game.alliance_parse(st, 0, "<ally decline 3>")["alliance_id"] == 3
    b = game.alliance_parse(st, 0, "<ally break 5>they lied</ally>")
    assert b["type"] == "alliance_break" and b["alliance_id"] == 5
    assert b["reason"] == "they lied"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
