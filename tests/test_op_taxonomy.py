from game_theory_llm.reasoning.op_taxonomy import (
    OP_TAGS, FAMILY_TAGS, BENCH_TAGS, BENCH_KNOWLEDGE, TRAINED_TAGS, overlap)

def test_tags_are_valid():
    for tags in list(FAMILY_TAGS.values()) + list(BENCH_TAGS.values()):
        assert set(tags) <= OP_TAGS
    assert TRAINED_TAGS <= OP_TAGS

def test_overlap_math():
    assert overlap(["backward_induction"], TRAINED_TAGS) == 1.0
    assert overlap(["constraint_satisfaction"], TRAINED_TAGS) == 0.0
    assert overlap(["backward_induction", "constraint_satisfaction"], TRAINED_TAGS) == 0.5
    assert overlap([], TRAINED_TAGS) == 0.0

def test_knowledge_flags():
    assert BENCH_KNOWLEDGE["mmlu_pro"] and BENCH_KNOWLEDGE["gpqa"]
    assert not BENCH_KNOWLEDGE["dyck"] and not BENCH_KNOWLEDGE["knights_knaves"]
