"""One Night Werewolf v2 feature-parity tests (spec §5.1).

Covers:
  * RandomPlayer matches at n in {3, 5, 7} all terminate with a valid winner;
  * whisper masking (a non-recipient's prompt never contains the whisper text);
  * a vote-bloc alliance can be both HONORED and BETRAYED (instrumented at the
    VOTE phase via judge_alliance);
  * the full role set / wake order / Doppelganger dynamic night slot / Tanner /
    Hunter / Minion / Drunk / Insomniac canon.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from game_theory_llm.play import run_match
from game_theory_llm.play.config import GameConfig
from game_theory_llm.play.games import OneNightWerewolf
from game_theory_llm.play.games.one_night_werewolf import (
    DOPPELGANGER,
    HUNTER,
    MINION,
    PHASE_DAY,
    PHASE_VOTE,
    ROBBER,
    SEER,
    TANNER,
    WEREWOLF,
)
from game_theory_llm.play.maps.onw_roles import (
    ALL_ROLES,
    ROLE_COUNTS_BY_N,
    WAKE_ORDER,
)
from game_theory_llm.play.players import RandomPlayer


VALID_WINNERS = ("village", "werewolves", "tanner", "nobody")


# --------------------------------------------------------------------------- helpers
def _fixed_deal(game: OneNightWerewolf, seats, center):
    """Build a state with a deterministic (un-shuffled) deal so role-canon
    tests can pin a known role to a known seat."""
    rng = random.Random(0)
    st = game.initial_state(rng)
    assert len(seats) == game.n_players and len(center) == 3
    st.initial_roles = list(seats)
    st.current_roles = list(seats)
    st.center_initial = list(center)
    st.center_current = list(center)
    st.doppel_copy = {}
    st.night_queue = game._build_night_queue(st.initial_roles)
    st.vote_queue = list(range(game.n_players))
    st.night_results = {}
    return st


def _state_at_day(game: OneNightWerewolf, seed: int = 0):
    """Drive a game through the night phase up to the DAY negotiation phase
    by always picking the first legal night action."""
    rng = random.Random(seed)
    st = game.initial_state(rng)
    guard = 0
    while st.phase != PHASE_DAY and guard < 200:
        guard += 1
        active = game.active_player(st)
        if active < 0:
            st = game.step(st, {"type": "advance_phase"})
            continue
        legal = game.legal_actions(st, active)
        st = game.step(st, legal[0])
    assert st.phase == PHASE_DAY, f"never reached DAY (phase={st.phase})"
    return st


# --------------------------------------------------------------------------- termination
def test_random_matches_terminate_at_3_5_7():
    for n in (3, 5, 7):
        for seed in range(6):
            game = OneNightWerewolf(n_players=n)
            players = [RandomPlayer(seed=seed * 100 + i) for i in range(n)]
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / f"onw_{n}_{seed}.jsonl"
                result = run_match(game, players, seed=seed, log_path=log_path,
                                   max_turns=800)
                ts = result.terminal_state
                assert game.is_terminal(ts), f"n={n} seed={seed} did not terminate"
                assert ts.winner_team in VALID_WINNERS, \
                    f"n={n} seed={seed} bad winner {ts.winner_team}"
                assert len(result.rewards) == n
                for r in result.rewards:
                    assert 0.0 <= r <= 1.0
                events = [json.loads(l) for l in log_path.read_text().splitlines()]
                assert events[-1]["type"] == "terminal"


def test_default_no_arg_constructor_is_5p():
    game = OneNightWerewolf()
    assert game.n_players == 5
    players = [RandomPlayer(seed=i) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        result = run_match(game, players, seed=1,
                           log_path=Path(tmp) / "m.jsonl", max_turns=800)
        assert result.terminal_state.winner_team in VALID_WINNERS


# --------------------------------------------------------------------------- deck / roles
def test_deck_size_is_n_plus_3():
    for n in range(3, 11):
        game = OneNightWerewolf(n_players=n)
        assert len(game._roles) == n + 3


def test_custom_roles_constructor():
    roles = ["Werewolf", "Werewolf", "Seer", "Robber", "Tanner",
             "Villager", "Villager", "Villager"]  # 8 == 5+3
    game = OneNightWerewolf(n_players=5, roles=roles)
    assert game._roles == roles


def test_custom_roles_wrong_length_rejected():
    import pytest
    with pytest.raises(ValueError):
        OneNightWerewolf(n_players=5, roles=["Werewolf"])  # too short


def test_wake_order_subset_of_all_roles():
    assert set(WAKE_ORDER).issubset(set(ALL_ROLES))


# --------------------------------------------------------------------------- whisper masking
def test_whisper_masking_recipient_vs_bystander():
    """A whisper's text must appear in the recipient's masked transcript but
    never in a non-recipient's prompt (the single masking point)."""
    game = OneNightWerewolf(n_players=5)
    st = _state_at_day(game)
    secret = "MEET_ME_AT_DAWN_xyzzy"
    # Seat 0 whispers to seat 1 only.
    action = game.parse_action(st, 0, f"<whisper to=1>{secret}</whisper>")
    assert action["type"] == "whisper" and action["to"] == [1]
    game.step(st, action)

    # Recipient (1) and sender (0) see the text; everyone else does not.
    assert secret in game.render_message_log(st, 0)
    assert secret in game.render_message_log(st, 1)
    for bystander in (2, 3, 4):
        log = game.render_message_log(st, bystander)
        assert secret not in log
        # And the full rendered prompt for a bystander also never leaks it.
        assert secret not in game.render_prompt(st, bystander)


def test_whisper_observation_routing_masks_bystanders():
    """nego_observations: recipients get full text, bystanders get metadata
    only (whisper_visibility default 'metadata')."""
    game = OneNightWerewolf(n_players=5)
    st = _state_at_day(game)
    secret = "TOPSECRET_qwerty"
    action = game.parse_action(st, 0, f"<whisper to=2>{secret}</whisper>")
    prev = st
    game.step(st, action)
    obs = game.observations(prev, st, action, actor=0)
    # Find the recipient obs (full) and bystander obs (meta).
    full = [o for o in obs if secret in str(o.payload)]
    meta = [o for o in obs if o.payload.get("type") == "message_meta"]
    assert full, "recipient observation with full text missing"
    # Recipient audience is {0, 2}.
    assert sorted(full[0].audience) == [0, 2]
    assert meta, "bystander metadata observation missing"
    for o in meta:
        assert secret not in str(o.payload)
        # bystanders 1,3,4 only.
        assert 0 not in o.audience and 2 not in o.audience


def test_public_say_is_broadcast():
    game = OneNightWerewolf(n_players=5)
    st = _state_at_day(game)
    msg = "I am the Seer, trust me"
    action = game.parse_action(st, 0, f"<say>{msg}</say>")
    prev = st
    game.step(st, action)
    obs = game.observations(prev, st, action, actor=0)
    assert len(obs) == 1
    assert sorted(obs[0].audience) == [0, 1, 2, 3, 4]
    # Everyone's transcript shows the public message.
    for seat in range(5):
        assert msg in game.render_message_log(st, seat)


# --------------------------------------------------------------------------- alliances honor/betray
def _drive_to_vote_with_bloc(game, vote_map):
    """Build a 5p state directly at the VOTE phase with a single ACTIVE
    vote_bloc alliance {seats 0,1} targeting seat 4, then apply ``vote_map``
    (voter -> target) and resolve. Returns the resolved state."""
    rng = random.Random(0)
    st = game.initial_state(rng)
    # Jump straight to VOTE (skip night/day for a controlled scenario).
    st.night_queue = []
    st.phase = PHASE_VOTE
    st.vote_queue = list(range(game.n_players))
    # Create + activate a vote_bloc alliance between 0 and 1 (target 4).
    propose = {"type": "alliance_propose", "to": [1], "kind": "vote_bloc",
               "terms": {"target": 4}}
    game.apply_alliance_action(st, 0, propose)
    aid = st.alli.next_id - 1
    game.apply_alliance_action(st, 1, {"type": "alliance_accept", "alliance_id": aid})
    assert st.alli.alliances[aid].status == "active"
    # Cast every vote, then resolve.
    for voter in range(game.n_players):
        st.votes[voter] = vote_map[voter]
    st = game._resolve(st)
    return st, aid


def test_vote_bloc_honored():
    """Both bloc members vote the agreed target -> two HONORED events."""
    game = OneNightWerewolf(n_players=5)
    votes = {0: 4, 1: 4, 2: 0, 3: 0, 4: 0}
    st, aid = _drive_to_vote_with_bloc(game, votes)
    honored = [e for e in st.alli.events if e.get("event") == "honored"
               and e["alliance_id"] == aid]
    betrayed = [e for e in st.alli.events if e.get("event") == "betrayed"
                and e["alliance_id"] == aid]
    assert len(honored) == 2, f"expected 2 honored, got {honored}"
    assert not betrayed


def test_vote_bloc_betrayed():
    """Seat 1 votes its own ally (seat 0) -> a BETRAYED event against seat 0."""
    game = OneNightWerewolf(n_players=5)
    votes = {0: 4, 1: 0, 2: 3, 3: 2, 4: 0}
    st, aid = _drive_to_vote_with_bloc(game, votes)
    betrayed = [e for e in st.alli.events if e.get("event") == "betrayed"
                and e["alliance_id"] == aid]
    assert betrayed, "expected a betrayed event"
    b = betrayed[0]
    assert b["actor"] == 1
    assert 0 in b["counterparty"]
    # Seat 0 still honored (voted the agreed target 4).
    honored = [e for e in st.alli.events if e.get("event") == "honored"
               and e["alliance_id"] == aid and e["actor"] == 0]
    assert honored


def test_alliance_summary_counts_in_terminal_log():
    """A full match where a vote-bloc forms and is partly betrayed.

    Under the per-alliance accounting contract, an alliance that has BOTH a
    honored and a betrayed judgement is classified into exactly ONE terminal
    bucket — betrayal dominates — so the single bloc counts as betrayed
    (n_betrayed == 1, n_honored == 0). The raw per-occasion tallies
    (honored_events / betrayed_events) still surface that both kinds of
    judgement happened, and the headline rates stay bounded in [0, 1]."""
    game = OneNightWerewolf(n_players=5)
    votes = {0: 4, 1: 0, 2: 4, 3: 4, 4: 0}
    st, aid = _drive_to_vote_with_bloc(game, votes)
    from game_theory_llm.play.alliances import alliance_summary
    summ = alliance_summary(st.alli)
    assert summ["n_accepted"] == 1
    # Betrayal dominates: the one bloc is counted as betrayed, not honored.
    assert summ["n_betrayed"] == 1
    assert summ["n_honored"] == 0
    assert summ["n_honored"] + summ["n_betrayed"] == summ["n_accepted"]
    # ...but both kinds of per-occasion judgement are preserved as diagnostics
    # (seat 0 voted the agreed target = honored occasion; others betrayed).
    assert summ["honored_events"] >= 1
    assert summ["betrayed_events"] >= 1
    # Headline rates are bounded.
    assert 0.0 <= summ.get("betrayal_rate", 0.0) <= 1.0


# --------------------------------------------------------------------------- role canon
def test_doppelganger_copies_and_acts_in_copied_slot():
    """Doppelganger copies a Seer and then gets a dynamic Seer night slot."""
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [DOPPELGANGER, SEER, WEREWOLF, "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    # Seat 0 is the Doppelganger (first in wake order); copy seat 1 (the Seer).
    active = game.active_player(st)
    assert active == 0 and st.initial_roles[0] == DOPPELGANGER
    st = game.step(st, {"type": "doppel_copy", "target": 1})
    assert st.doppel_copy[0] == SEER
    # A dynamic Seer slot for seat 0 must now exist in the night queue.
    assert (0, SEER) in st.night_queue, st.night_queue
    # Drive remaining night actions; seat 0 should get a Seer-style prompt.
    saw_doppel_seer = False
    guard = 0
    while st.phase != PHASE_DAY and guard < 100:
        guard += 1
        a = game.active_player(st)
        if a < 0:
            st = game.step(st, {"type": "advance_phase"})
            continue
        seat, role = st.night_queue[0]
        if seat == 0 and role == SEER:
            prompt = game.render_prompt(st, 0)
            assert "Seer" in prompt
            saw_doppel_seer = True
        st = game.step(st, game.legal_actions(st, a)[0])
    assert saw_doppel_seer


def test_doppelganger_werewolf_on_werewolf_team():
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [DOPPELGANGER, WEREWOLF, "Villager", "Villager",
                            "Villager"], ["Villager", "Villager", "Villager"])
    st = game.step(st, {"type": "doppel_copy", "target": 1})  # copy the wolf
    assert game._seat_team(st, 0) == "werewolf"


def test_tanner_wins_iff_tanner_dies():
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [TANNER, WEREWOLF, WEREWOLF, "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    st.night_queue = []
    st.phase = PHASE_VOTE
    st.vote_queue = []
    # Everyone votes the Tanner (seat 0).
    st.votes = {1: 0, 2: 0, 3: 0, 4: 0, 0: 1}
    st = game._resolve(st)
    assert st.winner_team == "tanner"
    rewards = game.rewards(st)
    assert rewards[0] == 1.0
    assert all(rewards[i] == 0.0 for i in range(1, 5))


def test_hunter_drags_vote_target_on_death():
    """When a Hunter is eliminated, their vote target is also eliminated."""
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [HUNTER, WEREWOLF, "Villager", "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    st.night_queue = []
    st.phase = PHASE_VOTE
    st.vote_queue = []
    # Hunter (0) is lynched and had voted the werewolf (1) -> wolf dies too.
    st.votes = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0}
    st = game._resolve(st)
    assert 0 in st.eliminated
    assert 1 in st.eliminated, "Hunter should drag down vote target (the wolf)"
    # A werewolf died -> village wins.
    assert st.winner_team == "village"


def test_minion_is_death_safe_and_on_wolf_team():
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [MINION, WEREWOLF, "Villager", "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    st.night_queue = []
    st.phase = PHASE_VOTE
    st.vote_queue = []
    # Minion (0) is lynched but no WEREWOLF card dies -> wolves win.
    st.votes = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0}
    st = game._resolve(st)
    assert st.winner_team == "werewolves", st.win_reason
    rewards = game.rewards(st)
    assert rewards[0] == 1.0   # Minion wins with the wolves.
    assert rewards[1] == 1.0   # Werewolf wins.


def test_drunk_swaps_blind_no_info():
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, ["Drunk", WEREWOLF, "Villager", "Villager", "Villager"],
                     ["Tanner", "Minion", "Seer"])
    # Drive to the Drunk's slot.
    guard = 0
    while st.phase != PHASE_DAY and guard < 100:
        guard += 1
        a = game.active_player(st)
        if a < 0:
            st = game.step(st, {"type": "advance_phase"})
            continue
        seat, role = st.night_queue[0]
        if role == "Drunk":
            before_center = list(st.center_current)
            st = game.step(st, {"type": "drunk_swap", "center": 0})
            # The drunk's result text must NOT reveal the new role.
            res = st.night_results[seat]
            assert "blind" in res.lower()
            assert before_center[0] not in res  # no role name leaked
            continue
        st = game.step(st, game.legal_actions(st, a)[0])


def test_insomniac_sees_final_card():
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, ["Insomniac", ROBBER, WEREWOLF, "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    guard = 0
    while st.phase != PHASE_DAY and guard < 100:
        guard += 1
        a = game.active_player(st)
        if a < 0:
            st = game.step(st, {"type": "advance_phase"})
            continue
        seat, role = st.night_queue[0]
        if role == "Insomniac":
            st = game.step(st, {"type": "insomniac_check"})
            res = st.night_results[seat]
            assert st.current_roles[seat] in res
            continue
        st = game.step(st, game.legal_actions(st, a)[0])


# --------------------------------------------------------------------------- night reveals
def test_night_peek_reveal_scoped_to_peeker():
    """A Seer's peek emits a reveal observation scoped to the peeker only."""
    game = OneNightWerewolf(n_players=5)
    st = _fixed_deal(game, [SEER, WEREWOLF, "Villager", "Villager", "Villager"],
                     ["Villager", "Villager", "Villager"])
    # Seat 0 is the Seer (wakes after wolves; but seat 1 wolf acks first).
    guard = 0
    while st.phase != PHASE_DAY and guard < 100:
        guard += 1
        a = game.active_player(st)
        if a < 0:
            st = game.step(st, {"type": "advance_phase"})
            continue
        seat, role = st.night_queue[0]
        if role == SEER:
            action = {"type": "seer_peek_player", "target": 1}
            prev = st
            game.step(st, action)
            obs = game.observations(prev, st, action, actor=seat)
            assert len(obs) == 1
            assert obs[0].audience == [seat]
            assert obs[0].payload["type"] == "reveal"
            break
        st = game.step(st, game.legal_actions(st, a)[0])


# --------------------------------------------------------------------------- observability
def test_god_view_and_snapshot_and_board():
    game = OneNightWerewolf(n_players=5)
    rng = random.Random(0)
    st = game.initial_state(rng)
    gv = game.god_view(st)
    assert "initial_roles" in gv and "current_roles" in gv
    assert len(gv["initial_roles"]) == 5
    snap = game.snapshot(st)
    assert "public" in snap and "hidden" in snap
    board_god = game.render_board(st, reveal="god")
    board_seat = game.render_board(st, reveal="seat0")
    # God reveal shows real role names; seat0 reveal hides others ("???").
    assert "Center:" in board_god
    assert "???" in board_seat
    assert "???" not in board_god


def test_messaging_off_skips_negotiation():
    """With messaging+alliances OFF the DAY phase has no chat actions but the
    match still terminates."""
    cfg = GameConfig(messaging=False, alliances=False, nego_rounds=1)
    game = OneNightWerewolf(n_players=5, config=cfg)
    players = [RandomPlayer(seed=i) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        result = run_match(game, players, seed=3,
                           log_path=Path(tmp) / "m.jsonl", max_turns=800)
        assert result.terminal_state.winner_team in VALID_WINNERS


if __name__ == "__main__":
    test_random_matches_terminate_at_3_5_7()
    test_whisper_masking_recipient_vs_bystander()
    test_vote_bloc_honored()
    test_vote_bloc_betrayed()
    print("ok")
