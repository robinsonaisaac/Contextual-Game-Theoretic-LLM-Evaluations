"""Rules, simultaneity, and measurement tests for the repeated public goods game.

Covers:
  * matches terminate after exactly ``rounds`` rounds under RandomPlayer, and
    the JSONL log carries a terminal record;
  * linear payoff arithmetic, and cumulative earnings == rewards;
  * SIMULTANEITY: an in-round contribution is routed to nobody (god-log only)
    and never appears in another seat's prompt; the resolved round IS public;
  * the action grammar (tag, bare integer, range check) and punishment parsing;
  * the punishment stage debits punisher and target at the configured rates;
  * the judge-free outcome measures: universal defection scores the 1/m
    efficiency floor, universal contribution scores 1.0, and the contribution
    decay curve is recorded per round;
  * the game refuses parameters that are not a social dilemma (m outside 1<m<n);
  * submission and speaking order rotate with the round.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from game_theory_llm.play import run_match
from game_theory_llm.play.base import ParseError
from game_theory_llm.play.config import GameConfig
from game_theory_llm.play.games import PublicGoods
from game_theory_llm.play.games.public_goods import (
    PH_CONTRIBUTE, PH_PUNISH, PH_TALK, PH_TERMINAL,
)
from game_theory_llm.play.players import RandomPlayer

NO_TALK = GameConfig(messaging=False, alliances=False, nego_rounds=0)


class FixedPlayer:
    """Contributes a constant amount; never talks."""

    def __init__(self, amount: int, name: str = "Fixed"):
        self.amount = amount
        self.name = name

    def act(self, game, state, idx):
        if state.phase == PH_TALK:
            return {"type": "pass_talk"}
        if state.phase == PH_PUNISH:
            return {"type": "punish", "points": {}}
        return {"type": "contribute", "amount": self.amount}

    def receive_observation(self, obs):
        return


def _run(game, players, seed=0, max_turns=800):
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "pg.jsonl"
        res = run_match(game, players, seed=seed, log_path=lp, max_turns=max_turns)
        events = [json.loads(x) for x in lp.read_text().splitlines()]
    return res, events


# --------------------------------------------------------------- termination
def test_terminates_after_exactly_rounds():
    for n in (2, 3, 5):
        g = PublicGoods(NO_TALK, n_players=n, rounds=4, endowment=10,
                        multiplier=1.6)
        res, events = _run(g, [RandomPlayer(seed=i) for i in range(n)])
        st = res.terminal_state
        assert g.is_terminal(st) and st.phase == PH_TERMINAL
        assert len(st.history) == 4
        assert [r["round"] for r in st.history] == [0, 1, 2, 3]
        assert events[-1]["type"] == "terminal"


def test_terminates_with_cheap_talk_enabled():
    g = PublicGoods(GameConfig(alliances=False, nego_rounds=1, msgs_per_slot=1),
                    n_players=5, rounds=3)
    res, events = _run(g, [RandomPlayer(seed=i) for i in range(5)])
    assert g.is_terminal(res.terminal_state)
    assert len(res.terminal_state.history) == 3
    assert any(e["type"] == "observation" and
               e["obs"].get("type") == "message" for e in events)


class TextPlayer:
    """Emits the tags an LLM would, exercising render -> text -> parse fully."""

    name = "Text"

    def __init__(self, amount: int, seed: int = 0):
        self.amount = amount
        self._rng = random.Random(seed)

    def act(self, game, state, idx):
        if state.phase == PH_TALK:
            return self._rng.choice([
                "<say>I will match whatever the group puts in.</say>",
                f"<whisper to={(idx + 1) % state.n_players}>let us both go high"
                "</whisper>",
                "<pass>",
            ])
        if state.phase == PH_PUNISH:
            return "<punish>none</punish>"
        return f"<contribute>{self.amount}</contribute>"

    def receive_observation(self, obs):
        return


def test_end_to_end_through_the_text_grammar():
    g = PublicGoods(GameConfig(alliances=False, nego_rounds=1, msgs_per_slot=1),
                    n_players=5, rounds=3, endowment=20, multiplier=2.0)
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "pg_text.jsonl"
        res = run_match(g, [TextPlayer(amount=8, seed=i) for i in range(5)],
                        seed=1, log_path=lp, max_turns=400)
        events = [json.loads(x) for x in lp.read_text().splitlines()]
    assert g.is_terminal(res.terminal_state)
    assert not [e for e in events if e["type"] == "parse_error"]
    assert not [e for e in events if e["type"] == "aborted"]
    # each round: keep 20-8=12, share 2.0*40/5=16  ->  28/round, 3 rounds
    assert res.rewards == pytest.approx([84.0] * 5)
    # No delivered observation ever carries an in-round contribution.
    for e in events:
        if e["type"] == "observation" and e["audience"]:
            assert e["obs"].get("type") != "contribution_private"
    # ...but the god log records every one of them.
    private = [e for e in events if e["type"] == "observation"
               and e["obs"].get("type") == "contribution_private"]
    assert len(private) == 15 and all(e["audience"] == [] for e in private)
    results = [e for e in events if e["type"] == "observation"
               and e["obs"].get("type") == "round_result"]
    assert len(results) == 3 and all(e["audience"] == [0, 1, 2, 3, 4]
                                     for e in results)


# ------------------------------------------------------------------ payoffs
def test_linear_payoff_arithmetic():
    g = PublicGoods(NO_TALK, n_players=5, rounds=1, endowment=20, multiplier=2.0)
    res, _ = _run(g, [FixedPlayer(a) for a in (0, 5, 10, 15, 20)])
    st = res.terminal_state
    rec = st.history[0]
    assert rec["total"] == 50
    assert rec["share"] == pytest.approx(2.0 * 50 / 5)          # 20.0
    # keep (E - c) + share
    assert rec["payoffs"] == pytest.approx([40.0, 35.0, 30.0, 25.0, 20.0])
    assert res.rewards == pytest.approx([40.0, 35.0, 30.0, 25.0, 20.0])
    assert st.winner == "P0"           # the free rider earns most


def test_earnings_accumulate_across_rounds():
    g = PublicGoods(NO_TALK, n_players=4, rounds=3, endowment=10, multiplier=2.0)
    res, _ = _run(g, [FixedPlayer(10) for _ in range(4)])
    # everyone contributes everything: payoff = 0 + 2*40/4 = 20 per round
    assert res.rewards == pytest.approx([60.0] * 4)


# ------------------------------------------------------------- simultaneity
def test_in_round_contribution_reaches_nobody_but_is_god_logged():
    g = PublicGoods(NO_TALK, n_players=5, rounds=2, endowment=20, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    first = st.submit_queue[0]
    action = {"type": "contribute", "amount": 13}
    g.step(st, action)
    obs = g.observations(st, st, action, actor=first)
    assert len(obs) == 1
    assert obs[0].audience == []                       # delivered to no seat
    assert obs[0].payload == {}
    assert obs[0].log["amount"] == 13                  # but fully auditable
    assert obs[0].log["player"] == first


def test_other_seats_prompt_does_not_show_the_pending_contribution():
    g = PublicGoods(NO_TALK, n_players=5, rounds=2, endowment=20, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    first = st.submit_queue[0]
    before = {seat: g.render_prompt(st, seat) for seat in range(5)}
    g.step(st, {"type": "contribute", "amount": 13})
    # A pending choice changes NOTHING another seat can see: the prompt is
    # byte-identical before and after, which is the strongest statement of
    # simultaneity available at this layer.
    for seat in range(5):
        if seat == first:
            continue
        after = g.render_prompt(st, seat)
        assert after == before[seat]
        assert "No rounds completed yet." in after


def test_resolved_round_is_public_to_everyone():
    g = PublicGoods(NO_TALK, n_players=3, rounds=1, endowment=10, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    for _ in range(3):
        action = {"type": "contribute", "amount": 4}
        g.step(st, action)
    obs = g.observations(st, st, action, actor=0)
    # the last mover's own choice (private) + the public round result
    assert len(obs) == 2
    assert obs[0].audience == [] and obs[0].log["type"] == "contribution_private"
    assert obs[1].audience == [0, 1, 2]
    assert obs[1].payload["type"] == "round_result"
    assert obs[1].payload["total"] == 12


# ------------------------------------------------------------------ parsing
def test_contribution_grammar():
    g = PublicGoods(NO_TALK, n_players=5, rounds=2, endowment=20, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    assert g.parse_action(st, 0, "<contribute>7</contribute>") == \
        {"type": "contribute", "amount": 7}
    assert g.parse_action(st, 0, "  12  ") == \
        {"type": "contribute", "amount": 12}          # bare integer tolerated
    with pytest.raises(ParseError, match="between 0 and 20"):
        g.parse_action(st, 0, "<contribute>25</contribute>")
    with pytest.raises(ParseError, match="between 0 and 20"):
        g.parse_action(st, 0, "<contribute>-3</contribute>")
    with pytest.raises(ParseError, match="expected"):
        g.parse_action(st, 0, "I will give a fair share to the group")


def test_punishment_grammar():
    g = PublicGoods(NO_TALK, n_players=5, rounds=1, endowment=20,
                    multiplier=2.0, punishment=True, max_punish_points=10)
    st = g.initial_state(random.Random(0))
    for _ in range(5):
        g.step(st, {"type": "contribute", "amount": 5})
    assert st.phase == PH_PUNISH
    seat = st.submit_queue[0]
    assert g.parse_action(st, seat, "<punish>none</punish>") == \
        {"type": "punish", "points": {}}
    other = (seat + 1) % 5
    parsed = g.parse_action(st, seat, f"<punish>P{other}:2</punish>")
    assert parsed["points"] == {other: 2}
    with pytest.raises(ParseError, match="cannot punish yourself"):
        g.parse_action(st, seat, f"<punish>P{seat}:1</punish>")
    with pytest.raises(ParseError, match="exceeds the limit"):
        g.parse_action(st, seat, f"<punish>P{other}:99</punish>")


def test_punishment_debits_punisher_and_target():
    g = PublicGoods(NO_TALK, n_players=3, rounds=1, endowment=10,
                    multiplier=2.0, punishment=True,
                    punish_cost=1, punish_impact=3)
    st = g.initial_state(random.Random(0))
    for _ in range(3):
        g.step(st, {"type": "contribute", "amount": 3})
    assert st.phase == PH_PUNISH
    base = list(st.earnings)
    order = list(st.submit_queue)
    punisher, target = order[0], order[1]
    g.step(st, {"type": "punish", "points": {target: 2}})
    for _ in range(2):
        g.step(st, {"type": "punish", "points": {}})
    assert st.earnings[punisher] == pytest.approx(base[punisher] - 2)
    assert st.earnings[target] == pytest.approx(base[target] - 6)
    assert g.is_terminal(st)


# -------------------------------------------------------- outcome measures
def test_universal_defection_hits_the_efficiency_floor():
    g = PublicGoods(NO_TALK, n_players=5, rounds=5, endowment=20, multiplier=2.0)
    res, _ = _run(g, [FixedPlayer(0) for _ in range(5)])
    st = res.terminal_state
    assert st.mean_contribution_rate == 0.0
    assert st.free_ride_rate == 1.0
    assert st.group_efficiency == pytest.approx(0.5)       # == 1/m
    assert st.defection_floor_efficiency == pytest.approx(0.5)


def test_universal_contribution_is_fully_efficient():
    g = PublicGoods(NO_TALK, n_players=5, rounds=5, endowment=20, multiplier=2.0)
    res, _ = _run(g, [FixedPlayer(20) for _ in range(5)])
    st = res.terminal_state
    assert st.mean_contribution_rate == 1.0
    assert st.free_ride_rate == 0.0
    assert st.group_efficiency == pytest.approx(1.0)


def test_per_round_contribution_curve_is_recorded():
    g = PublicGoods(NO_TALK, n_players=4, rounds=3, endowment=10, multiplier=2.0)
    res, _ = _run(g, [FixedPlayer(5) for _ in range(4)])
    st = res.terminal_state
    assert st.round_contribution_rates == [0.5, 0.5, 0.5]
    assert st.mean_contribution_rate == 0.5


# ---------------------------------------------------------------- validation
def test_rejects_parameters_that_are_not_a_social_dilemma():
    with pytest.raises(ValueError, match="social dilemma"):
        PublicGoods(NO_TALK, n_players=5, multiplier=0.9)     # no public good
    with pytest.raises(ValueError, match="social dilemma"):
        PublicGoods(NO_TALK, n_players=5, multiplier=5.0)     # no dilemma
    PublicGoods(NO_TALK, n_players=5, multiplier=2.0)         # fine


def test_submission_order_rotates_with_the_round():
    g = PublicGoods(NO_TALK, n_players=5, rounds=3, endowment=10, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    seen = []
    for _ in range(3):
        seen.append(list(st.submit_queue))
        for _ in range(5):
            g.step(st, {"type": "contribute", "amount": 1})
    assert seen[0][0] == 0 and seen[1][0] == 1 and seen[2][0] == 2
    assert all(sorted(o) == [0, 1, 2, 3, 4] for o in seen)


def test_phases_advance_talk_then_contribute():
    g = PublicGoods(GameConfig(alliances=False, nego_rounds=1, msgs_per_slot=1),
                    n_players=3, rounds=1, endowment=10, multiplier=2.0)
    st = g.initial_state(random.Random(0))
    assert st.phase == PH_TALK
    for _ in range(3):
        g.step(st, {"type": "pass_talk"})
    assert st.phase == PH_CONTRIBUTE
