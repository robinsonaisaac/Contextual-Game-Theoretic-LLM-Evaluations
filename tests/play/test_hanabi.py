"""Rules and masking tests for Hanabi (the pure-cooperation control game).

Covers:
  * matches at n in {2,3,4,5} terminate with a valid score under RandomPlayer;
  * the deck is the official 50 cards with the official rank multiplicities;
  * a seat NEVER sees its own hand, and the card drawn to replace a played or
    discarded card is not delivered to its owner (god-log only);
  * clues mark positive AND negative knowledge, cost a token, may not be empty,
    and may not be self-directed;
  * play/discard/fuse/clue-token economy, including the 5-completes-a-firework
    token refund and the discard-illegal-at-8 rule;
  * three fuses ends the match and scores 0 under the strict convention while
    the raw firework total is still recorded;
  * one final round after the deck empties, and identical rewards for all seats.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from game_theory_llm.play import run_match
from game_theory_llm.play.base import ParseError
from game_theory_llm.play.games import Hanabi
from game_theory_llm.play.games.hanabi import (
    COLORS, MAX_CLUES, N_FUSES, RANK_COUNTS, PH_TERMINAL, _card,
)
from game_theory_llm.play.players import RandomPlayer


def _fresh(n=5, seed=0, **kw):
    g = Hanabi(n_players=n, **kw)
    return g, g.initial_state(random.Random(seed))


# --------------------------------------------------------------- termination
def test_terminates_with_valid_score():
    for n in (2, 3, 4, 5):
        for seed in range(3):
            g = Hanabi(n_players=n)
            players = [RandomPlayer(seed=seed * 17 + i) for i in range(n)]
            with tempfile.TemporaryDirectory() as tmp:
                lp = Path(tmp) / f"h_{n}_{seed}.jsonl"
                res = run_match(g, players, seed=seed, log_path=lp, max_turns=400)
                events = [json.loads(x) for x in lp.read_text().splitlines()]
            st = res.terminal_state
            assert g.is_terminal(st), f"n={n} seed={seed} did not terminate"
            assert st.end_reason in ("bombed", "perfect", "deck_exhausted")
            assert 0 <= g.score(st) <= 25
            assert events[-1]["type"] == "terminal"
            # cooperative: one shared payoff
            assert len(set(res.rewards)) == 1


class TextPlayer:
    """Emits the same tags an LLM would, so the full render -> text -> parse
    path is exercised end to end (RandomPlayer returns pre-parsed dicts and
    silently skips ``parse_action``)."""

    name = "Text"

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def act(self, game, state, idx):
        a = self._rng.choice(game.legal_actions(state, idx))
        if a["type"] == "play":
            return f"<play>{a['index'] + 1}</play>"
        if a["type"] == "discard":
            return f"<discard>{a['index'] + 1}</discard>"
        return f"<clue>P{a['to']} {a['value']}</clue>"

    def receive_observation(self, obs):
        return


def test_end_to_end_through_the_text_grammar():
    g = Hanabi(n_players=5)
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "h_text.jsonl"
        res = run_match(g, [TextPlayer(seed=i) for i in range(5)],
                        seed=3, log_path=lp, max_turns=400)
        events = [json.loads(x) for x in lp.read_text().splitlines()]
    assert g.is_terminal(res.terminal_state)
    assert events[-1]["type"] == "terminal"
    # every emitted tag parsed on the first attempt
    assert not [e for e in events if e["type"] == "parse_error"]
    assert not [e for e in events if e["type"] == "aborted"]
    kinds = {e["type"] for e in events}
    assert {"match_start", "setup", "state_snapshot", "prompt", "action",
            "observation", "terminal"} <= kinds


# --------------------------------------------------------------------- deck
def test_official_deck_composition():
    g, st = _fresh()
    allcards = [c for h in st.hands for c in h] + st.deck
    assert len(allcards) == 50
    by_color = Counter(c["color"] for c in allcards)
    assert set(by_color) == set(COLORS)
    assert all(v == 10 for v in by_color.values())
    for color in COLORS:
        ranks = Counter(c["rank"] for c in allcards if c["color"] == color)
        assert dict(ranks) == RANK_COUNTS


def test_reduced_suit_variant_is_self_consistent():
    """n_colors is a measurement knob for escaping a score floor; the deck,
    board, scoring, and clue vocabulary must all shrink together."""
    g = Hanabi(n_players=3, n_colors=3)
    st = g.initial_state(random.Random(1))
    allcards = [c for h in st.hands for c in h] + st.deck
    assert len(allcards) == 30                      # 3 suits x 10
    assert set(c["color"] for c in allcards) == set(COLORS[:3])
    assert set(st.fireworks) == set(COLORS[:3])
    assert g.max_score == 15
    # a suit that is not in play is not a legal clue value
    st.hands[1] = [_card("red", 1)] * 5
    with pytest.raises(ParseError, match="clue value must be"):
        g.parse_action(st, 0, "<clue>P1 white</clue>")
    # and a perfect reduced game scores 1.0
    for c in COLORS[:3]:
        st.fireworks[c] = 5
    st.fireworks[COLORS[2]] = 4
    st.hands[0][0] = _card(COLORS[2], 5)
    st.to_move = 0
    g.step(st, {"type": "play", "index": 0})
    assert g.score(st) == 15 and g.rewards(st) == [1.0] * 3

    with pytest.raises(ValueError, match="n_colors"):
        Hanabi(n_players=3, n_colors=1)


def test_hand_size_by_player_count():
    assert len(Hanabi(n_players=2).initial_state(random.Random(0)).hands[0]) == 5
    assert len(Hanabi(n_players=3).initial_state(random.Random(0)).hands[0]) == 5
    assert len(Hanabi(n_players=4).initial_state(random.Random(0)).hands[0]) == 4
    assert len(Hanabi(n_players=5).initial_state(random.Random(0)).hands[0]) == 4


# ------------------------------------------------------------------ masking
def test_own_hand_is_hidden_but_others_are_visible():
    g, st = _fresh()
    prompt = g.render_prompt(st, 0)
    own = prompt.split("--- YOUR hand")[1].split("--- recent clues")[0] \
        .split("--- your options")[0]
    # No clues yet: every own slot is fully unknown.
    for i in range(g.hand_size):
        assert f"[{i + 1}] colour unknown, rank unknown" in own
    # ...and the seat's own concrete cards are not printed in that section.
    for c in st.hands[0]:
        assert f"{c['color']}{c['rank']}" not in own
    # Other seats' hands are fully visible.
    others = prompt.split("--- other players' hands")[1].split("--- YOUR hand")[0]
    for c in st.hands[1]:
        assert f"{c['color']}{c['rank']}" in others


def test_drawn_replacement_card_is_not_delivered_to_owner():
    g, st = _fresh()
    st.deck.append(_card("red", 5))          # known top card
    action = {"type": "play", "index": 0}
    g.step(st, action)
    obs = g.observations(st, st, action, actor=0)
    assert len(obs) == 1
    assert obs[0].audience == list(range(g.n_players))
    assert "drew" not in obs[0].payload          # delivered payload is clean
    assert "_drew" not in obs[0].payload
    assert obs[0].log["drew"] == {"color": "red", "rank": 5}   # god log has it


# -------------------------------------------------------------------- clues
def test_clue_marks_positive_and_negative_knowledge_and_costs_a_token():
    g, st = _fresh()
    st.hands[1] = [_card("red", 1), _card("blue", 2), _card("red", 3),
                   _card("green", 4)]
    st.knowledge[1] = [dict(color=None, rank=None, not_colors=[], not_ranks=[])
                       for _ in range(4)]
    before = st.clue_tokens
    g.step(st, {"type": "clue", "to": 1, "kind": "color", "value": "red"})
    assert st.clue_tokens == before - 1
    k = st.knowledge[1]
    assert k[0]["color"] == "red" and k[2]["color"] == "red"
    assert k[1]["color"] is None and "red" in k[1]["not_colors"]
    assert "red" in k[3]["not_colors"]


def test_rank_clue_marks_ranks():
    g, st = _fresh()
    st.hands[1] = [_card("red", 3), _card("blue", 2), _card("green", 3),
                   _card("white", 4)]
    st.knowledge[1] = [dict(color=None, rank=None, not_colors=[], not_ranks=[])
                       for _ in range(4)]
    g.step(st, {"type": "clue", "to": 1, "kind": "rank", "value": 3})
    k = st.knowledge[1]
    assert k[0]["rank"] == 3 and k[2]["rank"] == 3
    assert 3 in k[1]["not_ranks"] and 3 in k[3]["not_ranks"]


def test_empty_clue_and_self_clue_are_rejected():
    g, st = _fresh()
    st.hands[1] = [_card("red", 1)] * 4
    with pytest.raises(ParseError, match="empty clue"):
        g.parse_action(st, 0, "<clue>P1 blue</clue>")
    with pytest.raises(ParseError, match="cannot clue yourself"):
        g.parse_action(st, 0, "<clue>P0 red</clue>")


def test_clue_illegal_at_zero_tokens():
    g, st = _fresh()
    st.hands[1] = [_card("red", 1)] * 4
    st.clue_tokens = 0
    with pytest.raises(ParseError, match="no clue tokens"):
        g.parse_action(st, 0, "<clue>P1 red</clue>")
    assert not any(a["type"] == "clue" for a in g.legal_actions(st, 0))


# ------------------------------------------------------------ play / discard
def test_correct_play_advances_firework_and_wrong_play_spends_a_fuse():
    g, st = _fresh()
    st.hands[0][0] = _card("red", 1)
    g.step(st, {"type": "play", "index": 0})
    assert st.fireworks["red"] == 1
    assert st.fuse_tokens == N_FUSES

    st.to_move = 0                            # step() acts for state.to_move
    st.hands[0][0] = _card("red", 5)          # not the next rank
    g.step(st, {"type": "play", "index": 0})
    assert st.fireworks["red"] == 1
    assert st.fuse_tokens == N_FUSES - 1
    assert any(c == {"color": "red", "rank": 5} for c in st.discards)


def test_completing_a_firework_refunds_a_clue_token():
    g, st = _fresh()
    st.fireworks["red"] = 4
    st.clue_tokens = 3
    st.hands[0][0] = _card("red", 5)
    g.step(st, {"type": "play", "index": 0})
    assert st.fireworks["red"] == 5
    assert st.clue_tokens == 4


def test_discard_illegal_at_max_clue_tokens():
    g, st = _fresh()
    st.clue_tokens = MAX_CLUES
    with pytest.raises(ParseError, match="cannot discard"):
        g.parse_action(st, 0, "<discard>1</discard>")
    assert not any(a["type"] == "discard" for a in g.legal_actions(st, 0))
    st.clue_tokens = MAX_CLUES - 1
    assert g.parse_action(st, 0, "<discard>1</discard>") == \
        {"type": "discard", "index": 0}


def test_discard_returns_a_clue_token():
    g, st = _fresh()
    st.clue_tokens = 2
    g.step(st, {"type": "discard", "index": 0})
    assert st.clue_tokens == 3
    assert len(st.discards) == 1


# ----------------------------------------------------------------- endgame
def test_three_fuses_ends_the_match_and_scores_zero_when_strict():
    g, st = _fresh()
    st.fireworks["red"] = 3          # some real progress on the board
    st.fuse_tokens = 1
    st.hands[0][0] = _card("blue", 5)     # guaranteed misplay
    g.step(st, {"type": "play", "index": 0})
    assert st.phase == PH_TERMINAL and st.end_reason == "bombed"
    assert st.fireworks_score == 3        # raw total preserved for analysis
    assert g.score(st) == 0               # strict convention
    assert g.rewards(st) == [0.0] * g.n_players

    lenient = Hanabi(n_players=5, strict_bombs=False)
    assert lenient.score(st) == 3


def test_perfect_game_terminates_at_25():
    g, st = _fresh()
    for c in COLORS:
        st.fireworks[c] = 5
    st.fireworks["white"] = 4
    st.hands[0][0] = _card("white", 5)
    g.step(st, {"type": "play", "index": 0})
    assert st.end_reason == "perfect"
    assert g.score(st) == 25
    assert g.rewards(st) == [1.0] * g.n_players


def test_empty_deck_gives_every_seat_exactly_one_more_turn():
    g, st = _fresh(n=4)
    st.deck = []
    st.clue_tokens = 4
    g.step(st, {"type": "discard", "index": 0})
    assert st.final_turns_left == g.n_players
    for _ in range(g.n_players):
        assert not g.is_terminal(st)
        g.step(st, {"type": "discard", "index": 0})
    assert g.is_terminal(st) and st.end_reason == "deck_exhausted"


# ------------------------------------------------------------------ parsing
def test_action_grammar_round_trip():
    g, st = _fresh()
    st.hands[1] = [_card("red", 1), _card("blue", 2), _card("green", 3),
                   _card("white", 4)]
    assert g.parse_action(st, 0, "<play>2</play>") == {"type": "play", "index": 1}
    assert g.parse_action(st, 0, "<clue>P1 blue</clue>") == \
        {"type": "clue", "to": 1, "kind": "color", "value": "blue"}
    assert g.parse_action(st, 0, "<clue>P1 3</clue>") == \
        {"type": "clue", "to": 1, "kind": "rank", "value": 3}
    assert g.parse_action(st, 0, "<clue to=1>red</clue>") == \
        {"type": "clue", "to": 1, "kind": "color", "value": "red"}
    with pytest.raises(ParseError, match="slot must be"):
        g.parse_action(st, 0, "<play>9</play>")
    with pytest.raises(ParseError, match="expected"):
        g.parse_action(st, 0, "I think I will play the red one")
