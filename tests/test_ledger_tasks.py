import pytest
from game_theory_llm.reasoning.ledger_protocol import (
    Answer, skeleton_claims, score_claims, render_canonical, parse_canonical)
from game_theory_llm.reasoning.ledger_tasks import get_generator

# Grows as families land: Task 2 = trees; Task 3 += register_machine, graph_search;
# Task 4 += forward_chain, object_tracking, scheduling.
IMPLEMENTED = ["trees", "register_machine", "graph_search"]


@pytest.mark.parametrize("family", IMPLEMENTED)
@pytest.mark.parametrize("horizon", [12, 24, 48])
def test_gold_skeleton_scores_perfect(family, horizon):
    task = get_generator(family)(seed=7, horizon=horizon)
    notes, ans = skeleton_claims(task.events)
    s = score_claims(notes, ans, task.gold_facts, task.gold_answer)
    assert s["precision"] == 1.0 and s["recall"] == 1.0
    assert s["answer_correct"] is True
    assert isinstance(task.events[-1], Answer)
    assert task.horizon == len(task.gold_facts)


@pytest.mark.parametrize("family", IMPLEMENTED)
def test_deterministic(family):
    a = get_generator(family)(seed=3, horizon=30)
    b = get_generator(family)(seed=3, horizon=30)
    assert a.gold_answer == b.gold_answer and a.gold_facts == b.gold_facts
    assert render_canonical(a) == render_canonical(b)


@pytest.mark.parametrize("family", IMPLEMENTED)
def test_canonical_roundtrip(family):
    task = get_generator(family)(seed=11, horizon=20)
    assert parse_canonical(render_canonical(task)) == skeleton_claims(task.events)
