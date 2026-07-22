"""tests/play/test_runner_parse_policy.py — retry x3 then void, no fallback."""
import json
from pathlib import Path

import pytest

from game_theory_llm.play.games import OneNightWerewolf
from game_theory_llm.play.players.random import RandomPlayer
from game_theory_llm.play.runner import run_match


class ScriptedPlayer:
    """Returns canned strings; records observations it receives."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.name = "Scripted"
        self.seen = []
        self.n_act_calls = 0

    def act(self, game, state, player_idx):
        self.n_act_calls += 1
        return self.replies.pop(0) if self.replies else "garbage"

    def receive_observation(self, obs):
        self.seen.append(obs)


def _recs(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _run(tmp_path, seat0):
    game = OneNightWerewolf(n_players=5)
    players = [seat0] + [RandomPlayer(seed=i) for i in range(1, 5)]
    res = run_match(game, players, seed=1, log_path=tmp_path / "m.jsonl")
    return res, _recs(tmp_path / "m.jsonl")


def test_four_garbage_attempts_void_the_match(tmp_path):
    sp = ScriptedPlayer([])  # always garbage
    res, recs = _run(tmp_path, sp)
    parse_errors = [r for r in recs if r["type"] == "parse_error"]
    aborted = [r for r in recs if r["type"] == "aborted"]
    assert sp.n_act_calls == 4             # 1 initial + 3 informed retries
    assert len(parse_errors) == 3          # retries 1..3 logged
    assert len(aborted) == 1
    a = aborted[0]
    assert a["reason"] == "unparseable_output"
    assert a["player"] == 0
    assert a["model"] == "Scripted"
    assert "last_error" in a and "last_raw" in a
    assert not [r for r in recs if r["type"] == "terminal"]
    assert not [r for r in recs if r["type"] == "fallback"]
    assert res.metadata["aborted"] is True
    assert res.metadata["aborted_player"] == 0
    assert res.rewards == [0.0] * 5


def test_recovery_within_retries_completes(tmp_path):
    # ONW seat 0 (a werewolf-deck role) first acts in the night phase; two
    # garbage replies then defer to random-legal via a real parseable reply
    # is game-specific, so instead assert: garbage x2 then the runner's
    # third call gets a pre-parsed legal action (dict passthrough).
    class RecoveringPlayer(ScriptedPlayer):
        def act(self, game, state, player_idx):
            if self.replies:
                return self.replies.pop(0)
            legal = game.legal_actions(state, player_idx)
            return legal[0] if legal else {"type": "noop"}

    res, recs = _run(tmp_path, RecoveringPlayer(["garbage one", "garbage two"]))
    assert not [r for r in recs if r["type"] == "aborted"]
    assert [r for r in recs if r["type"] == "terminal"]
    assert res.metadata.get("aborted") is not True


def test_parse_error_observation_delivered_before_retry(tmp_path):
    sp = ScriptedPlayer([])
    _run(tmp_path, sp)
    pe = [o for o in sp.seen if o.get("type") == "parse_error"]
    assert len(pe) == 3 and all("error" in o for o in pe)


def test_llm_player_renders_corrective_line():
    from game_theory_llm.play.players.llm import LLMPlayer
    p = LLMPlayer.__new__(LLMPlayer)          # no client needed
    p.history = []
    p.max_history = 40
    p.receive_observation({"type": "parse_error", "error": "expected <vote>"})
    joined = "\n".join(str(h) for h in p.history)
    assert "could not be parsed" in joined and "expected <vote>" in joined


def test_steered_llm_player_renders_corrective_line():
    from game_theory_llm.play.players.steered_llm import SteeredLLMPlayer
    p = SteeredLLMPlayer.__new__(SteeredLLMPlayer)  # no Modal client needed
    p.history = []
    p.max_history = 40
    p.receive_observation({"type": "parse_error", "error": "expected <vote>"})
    joined = "\n".join(str(h) for h in p.history)
    assert "could not be parsed" in joined and "expected <vote>" in joined
