"""Tests for depth-scaled non-game eval generators (Task 2)."""
from game_theory_llm.reasoning.eval_gen import dyck, prontoqa


def test_dyck_closers_are_correct_and_depth_scaled():
    p = dyck(seed=1, depth=4)
    assert p["depth"] == 4
    assert set(p["answer"]) <= set(")]}>")
    assert len(p["answer"]) == 4           # one closer per open
    # the closers must correctly mirror the opens stated in the prompt
    import re
    seq = re.search(r"Input:\s*([(){}\[\]<>]+)", p["prompt"]).group(1)
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    assert p["answer"] == "".join(pairs[c] for c in reversed(seq))


def test_prontoqa_label_and_varies():
    labels = {prontoqa(seed=s, depth=3)["answer"] for s in range(20)}
    assert labels == {"True", "False"} or len(labels) == 2   # both labels occur
    p = prontoqa(seed=1, depth=3)
    assert p["answer"] in {"True", "False"} and p["depth"] == 3


def test_countdown_left_to_right_max():
    import re, itertools
    from game_theory_llm.reasoning.eval_gen import countdown
    p = countdown(seed=5, depth=3)
    nums = [int(x) for x in re.search(r"Numbers: \[([0-9,\s]+)\]", p["prompt"]).group(1).split(",")]
    best = None
    for ops in itertools.product("+-*", repeat=len(nums)-1):
        v = nums[0]
        for op, x in zip(ops, nums[1:]):
            v = v+x if op=="+" else v-x if op=="-" else v*x
        best = v if best is None else max(best, v)
    assert p["answer"] == best

def test_ordering_position():
    import re
    from game_theory_llm.reasoning.eval_gen import ordering
    p = ordering(seed=5, depth=4)
    assert 1 <= p["answer"] <= 5

def test_boolean_eval_label():
    from game_theory_llm.reasoning.eval_gen import boolean_eval
    assert boolean_eval(seed=5, depth=3)["answer"] in (0, 1)

def test_knights_unique_and_labeled():
    from game_theory_llm.reasoning.eval_gen import knights_knaves
    p = knights_knaves(seed=5, depth=4)   # internal assert guarantees uniqueness
    assert p["answer"] >= 1 and "nested_belief" in p["op_tags"]
