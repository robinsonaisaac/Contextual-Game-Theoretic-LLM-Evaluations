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
