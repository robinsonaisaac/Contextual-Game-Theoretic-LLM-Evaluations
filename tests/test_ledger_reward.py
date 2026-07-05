# tests/test_ledger_reward.py
from game_theory_llm.reasoning.ledger_protocol import parse_canonical, score_claims


def _reward(text, gold_facts, gold_answer):
    notes, ans = parse_canonical(text)
    s = score_claims(notes, ans, gold_facts, gold_answer)
    return float(s["answer_correct"]) + 0.5 * s["ledger_reward"]


def test_perfect_completion_scores_1_5():
    text = "LEDGER\nNOTE: a = 1\nNOTE: b = 2\nANSWER: 3"
    assert _reward(text, {"a": "1", "b": "2"}, "3") == 1.5


def test_wrong_everything_scores_0():
    text = "LEDGER\nNOTE: a = 9\nANSWER: 7"
    assert _reward(text, {"a": "1", "b": "2"}, "3") == 0.0


def test_right_answer_partial_ledger_between_0_and_1_5():
    text = "LEDGER\nNOTE: a = 1\nANSWER: 3"     # answer right, half the notes
    r = _reward(text, {"a": "1", "b": "2"}, "3")
    assert 1.0 < r < 1.5


def test_junk_notes_collapse_precision():
    """Spraying many wrong notes drives precision down → ledger_reward → 0."""
    junk = "\n".join(f"NOTE: x{i} = {i}" for i in range(100))
    text = f"LEDGER\n{junk}\nNOTE: a = 1\nANSWER: 3"
    r = _reward(text, {"a": "1"}, "3")
    # answer correct = 1.0; ledger_reward = precision*recall = (1/101)*1.0 ≈ 0.0099
    assert r < 1.1, f"spray guard failed, got {r}"
    assert r > 1.0, f"answer_correct should still contribute, got {r}"


def test_no_answer_no_notes_scores_0():
    text = "LEDGER"
    assert _reward(text, {"a": "1"}, "3") == 0.0
