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
    # PER-ALLIANCE classification: alliance #0 had BOTH an honored and a
    # betrayed judgement event. Betrayal dominates, so it is counted as
    # exactly ONE betrayed alliance and ZERO honored alliances (B6/M3/M4).
    assert s["n_honored"] == 0
    assert s["n_betrayed"] == 1
    # No alliance is double-counted: the headline honour/betray buckets sum to
    # the number of accepted (active) alliances.
    assert s["n_honored"] + s["n_betrayed"] == s["n_accepted"]
    # Raw per-occasion judgement counts survive as diagnostics.
    assert s["honored_events"] == 1
    assert s["betrayed_events"] == 1

    per = s["per_player"]
    assert per["0"]["proposed"] == 1
    assert per["0"]["honored"] == 1       # per-occasion attribution unchanged
    assert per["1"]["accepted"] == 1
    assert per["1"]["betrayed"] == 1
    assert per["0"]["betrayed_against"] == 1
    assert per["2"]["proposed"] == 1


def test_alliance_summary_empty_state():
    assert alliance_summary(None)["n_proposed"] == 0
    assert alliance_summary(AllianceState())["n_betrayed"] == 0


def _rates(s: dict):
    """Compute the §4 derived rates the way metrics.load_match does."""
    def r(num, den):
        return (num / den) if den else None
    return {
        "formation": r(s["n_accepted"], s["n_proposed"]),
        "honour": r(s["n_honored"], s["n_accepted"]),
        "betrayal": r(s["n_betrayed"], s["n_accepted"]),
    }


def test_rates_bounded_and_no_double_count():
    """B6/M3/M4: an alliance with MANY per-occasion honored/betrayed events
    must still be counted as exactly one terminal bucket; rates stay in
    [0, 1] and no alliance is both honored and betrayed."""
    alli = AllianceState()
    # Alliance #0: active, honoured several times then betrayed once.
    a0 = _make(alli, proposer=0, members=[0, 1], kind="pact")
    a0.status = "active"; a0.accepted_turn = 1
    alli.events.append(new_event("propose", a0, turn=0, actor=0, counterparty=[1]))
    alli.events.append(new_event("accept", a0, turn=1, actor=1, counterparty=[0]))
    for t in range(2, 7):  # five honour occasions
        alli.events.append(new_event("honored", a0, turn=t, actor=0, counterparty=[1]))
    for t in range(7, 10):  # three betrayal occasions on the SAME alliance
        alli.events.append(new_event("betrayed", a0, turn=t, actor=1, counterparty=[0]))

    # Alliance #1: active, only ever honoured.
    a1 = _make(alli, proposer=2, members=[2, 3], kind="truce")
    a1.status = "active"; a1.accepted_turn = 1
    alli.events.append(new_event("propose", a1, turn=0, actor=2, counterparty=[3]))
    alli.events.append(new_event("accept", a1, turn=1, actor=3, counterparty=[2]))
    alli.events.append(new_event("honored", a1, turn=4, actor=2, counterparty=[3]))

    s = alliance_summary(alli)
    # Two accepted alliances; #0 betrayed (betrayal dominates), #1 honoured.
    assert s["n_accepted"] == 2
    assert s["n_betrayed"] == 1
    assert s["n_honored"] == 1
    # No alliance is in BOTH buckets.
    assert s["n_honored"] + s["n_betrayed"] == s["n_accepted"]
    # Raw per-occasion diagnostics are preserved and (here) exceed the bucket.
    assert s["honored_events"] == 6
    assert s["betrayed_events"] == 3

    rates = _rates(s)
    for name, v in rates.items():
        assert v is not None and 0.0 <= v <= 1.0, f"{name} rate {v} out of [0,1]"
    assert rates["honour"] == 0.5 and rates["betrayal"] == 0.5
    assert rates["formation"] == 1.0


def test_summary_from_events_matches_alliance_summary():
    """metrics.summary_from_alliance_events must agree with alliance_summary on
    the same trail (so log-replay rates equal the live terminal rates)."""
    from game_theory_llm.play.alliances import summary_from_alliance_events

    alli = AllianceState()
    a0 = _make(alli, proposer=0, members=[0, 1], kind="pact")
    a0.status = "active"; a0.accepted_turn = 1
    alli.events.append(new_event("propose", a0, turn=0, actor=0, counterparty=[1]))
    alli.events.append(new_event("accept", a0, turn=1, actor=1, counterparty=[0]))
    alli.events.append(new_event("betrayed", a0, turn=3, actor=1, counterparty=[0]))

    live = alliance_summary(alli)
    replay = summary_from_alliance_events(list(alli.events))
    for k in ("n_proposed", "n_accepted", "n_declined", "n_broken",
              "n_honored", "n_betrayed"):
        assert live[k] == replay[k], f"{k}: live {live[k]} != replay {replay[k]}"


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


# --------------------------------------------------- end-to-end (real game)
def test_secret_hitler_match_rates_bounded_and_logged():
    """Run a real Secret Hitler match (messaging on) and assert:
      * the terminal alliance_summary rates are all <= 1.0 (B6/M3/M4 fixed),
      * no alliance is counted as both honored and betrayed,
      * the JSONL log carries top-level ``alliance_event`` records for the
        honored AND betrayed judgements the runner now emits (B7), and
      * the bounded per-alliance buckets sit below the raw per-occasion
        diagnostics (proof the redesign actually deduped)."""
    import json
    import tempfile
    from pathlib import Path

    from game_theory_llm.play import run_match
    from game_theory_llm.play.games import SecretHitler
    from game_theory_llm.play.players import RandomPlayer

    # SecretHitler has messaging on by default (GameConfig.messaging=True),
    # so RandomPlayers exercise the full propose/accept/break + honour/betray
    # lifecycle. Scan a few seeds to find one that produced both judgement
    # kinds (the random walk reaches them readily).
    chosen = None
    for seed in range(8):
        g = SecretHitler(n_players=5)
        players = [RandomPlayer(seed=seed * 13 + i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / f"sh_{seed}.jsonl"
            run_match(g, players, seed=seed, log_path=lp, max_turns=1500)
            events = [json.loads(line) for line in lp.read_text().splitlines()]
        top = [e for e in events if e.get("type") == "alliance_event"]
        kinds = {e.get("event") for e in top}
        if "honored" in kinds and "betrayed" in kinds:
            chosen = (events, top)
            break
    assert chosen is not None, \
        "no seed produced both honored and betrayed alliance_event records"
    events, top = chosen

    # B7: top-level alliance_event log records for the judgement events.
    assert any(e.get("event") == "honored" for e in top), \
        "no top-level honored alliance_event records in the JSONL log"
    assert any(e.get("event") == "betrayed" for e in top), \
        "no top-level betrayed alliance_event records in the JSONL log"

    summary = events[-1]["alliance_summary"]
    n_acc = summary["n_accepted"]
    n_prop = summary["n_proposed"]
    # Bounded per-alliance accounting.
    assert summary["n_honored"] + summary["n_betrayed"] == n_acc, \
        "honored + betrayed buckets must partition the accepted alliances"
    assert n_acc <= n_prop

    def rate(num, den):
        return (num / den) if den else 0.0

    formation = rate(n_acc, n_prop)
    honour = rate(summary["n_honored"], n_acc)
    betrayal = rate(summary["n_betrayed"], n_acc)
    for name, v in (("formation", formation), ("honour", honour),
                    ("betrayal", betrayal)):
        assert 0.0 <= v <= 1.0, f"{name} rate {v} exceeds [0, 1]"

    # The redesign deduped: raw per-occasion judgement counts dominate the
    # bounded buckets (this match has many honour occasions per alliance).
    assert summary["honored_events"] >= summary["n_honored"]
    assert summary["betrayed_events"] >= summary["n_betrayed"]


def test_secret_hitler_match_metrics_replay_matches_terminal():
    """metrics.load_match (log replay) must reproduce the terminal
    alliance_summary's bounded buckets and rates."""
    import json
    import tempfile
    from pathlib import Path

    from game_theory_llm.play import run_match
    from game_theory_llm.play.games import SecretHitler
    from game_theory_llm.play.players import RandomPlayer
    from game_theory_llm.play.metrics import load_match

    g = SecretHitler(n_players=5)
    players = [RandomPlayer(seed=13 + i) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "sh_replay.jsonl"
        run_match(g, players, seed=1, log_path=lp, max_turns=1500)
        events = [json.loads(line) for line in lp.read_text().splitlines()]
        m = load_match(str(lp))

    term_summary = events[-1]["alliance_summary"]
    for r in m["rates"].values():
        if r is not None:
            assert 0.0 <= r <= 1.0, f"replayed rate {r} out of [0, 1]"
    # The headline buckets agree between live terminal and log replay.
    assert m["alliances"]["n_accepted"] == term_summary["n_accepted"]
    assert m["alliances"]["n_honored"] == term_summary["n_honored"]
    assert m["alliances"]["n_betrayed"] == term_summary["n_betrayed"]
    # first_betrayal_turn is recoverable from the logged alliance_event records.
    if term_summary["n_betrayed"] > 0:
        assert m["first_betrayal_turn"] is not None


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
