import random
from game_theory_llm.reasoning.ledger_protocol import (
    Answer, Correction, LedgerTask, Note, Restate,
    render_canonical, parse_canonical, skeleton_claims, score_claims, inject_recovery,
)


def _task():
    events = [
        Note("a", "1", []),
        Note("b", "2", ["a"]),
        Note("c", "3", ["a", "b"]),
        Answer("3"),
    ]
    return LedgerTask("demo", "demo_1", "prompt body", "3",
                      {"a": "1", "b": "2", "c": "3"}, events, horizon=3)


def test_gold_skeleton_scores_perfect():
    t = _task()
    notes, ans = skeleton_claims(t.events)
    s = score_claims(notes, ans, t.gold_facts, t.gold_answer)
    assert s["precision"] == 1.0 and s["recall"] == 1.0
    assert s["answer_correct"] is True and s["ledger_reward"] == 1.0


def test_render_parse_roundtrip():
    t = _task()
    notes, ans = parse_canonical(render_canonical(t))
    assert notes == [("a", "1"), ("b", "2"), ("c", "3")]
    assert ans == "3"


def test_render_has_grammar_tokens():
    text = render_canonical(_task())
    assert text.startswith("LEDGER")
    assert "NEXT:" in text and "NOTE:" in text and "ANSWER: 3" in text


def test_spray_guard_collapses_precision():
    gold = {"a": "1", "b": "2", "c": "3"}
    notes = [("a", "1"), ("b", "2"), ("c", "3")] + [(f"j{i}", "9") for i in range(30)]
    s = score_claims(notes, "3", gold, "3")
    assert s["recall"] == 1.0
    assert s["precision"] < 0.2 and s["ledger_reward"] < 0.2


def test_correction_yields_right_value():
    text = "LEDGER\nNOTE: a = 5 (was 4) <- x\nANSWER: 5"
    notes, ans = parse_canonical(text)
    assert notes == [("a", "5")] and ans == "5"


def test_inject_recovery_adds_restate_and_correction():
    t = _task()
    ev = inject_recovery(t.events, random.Random(0))
    assert any(isinstance(e, Restate) for e in ev)
    assert any(isinstance(e, Correction) for e in ev)
    # after recovery the final claimed value for the corrupted key is correct again
    final = {}
    for e in ev:
        if isinstance(e, Note):
            final[e.key] = e.value
        elif isinstance(e, Correction):
            final[e.key] = e.right_value
    assert final == t.gold_facts
    assert isinstance(ev[-1], Answer) and ev[-1].value == "3"
