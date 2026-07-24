"""Tests for the game-theory reasoning data generator + verifier."""
import random
import re

from game_theory_llm.reasoning.gametree import (
    Node, minimax, _gen_tree, gen_game_tree_problem, gen_game_tree_sft,
    gen_level_k_problem,
)


def _brute(node, mx):
    if node.is_leaf:
        return node.value
    vs = [_brute(c, not mx) for _, c in node.children]
    return max(vs) if mx else min(vs)


def test_minimax_hand_tree():
    t = Node(False, children=[
        ("A", Node(False, children=[("C", Node(True, value=3)), ("D", Node(True, value=-2))])),
        ("B", Node(False, children=[("E", Node(True, value=5)), ("F", Node(True, value=1))])),
    ])
    assert minimax(t, True) == 1   # MAX{ MIN{3,-2}, MIN{5,1} } = MAX{-2,1} = 1


def test_minimax_matches_bruteforce_random():
    for s in range(150):
        rng = random.Random(s)
        root = _gen_tree(rng, depth=rng.randint(1, 6), branching=rng.choice([2, 3]),
                         leaf_lo=-9, leaf_hi=9)
        assert minimax(root, True) == _brute(root, True)


def test_problem_answer_is_verifiable_and_varies():
    answers = {gen_game_tree_problem(seed=s, depth=4)["answer"] for s in range(20)}
    assert len(answers) > 1            # not a constant the model could guess


def test_gold_cot_ends_with_correct_answer():
    for s in range(30):
        for d in (2, 3, 4):
            p = gen_game_tree_sft(seed=s, depth=d)
            m = re.search(r"<answer>(-?\d+)</answer>", p["completion"])
            assert m and int(m.group(1)) == p["answer"]


def test_level_k_depth_scaling():
    # p=2/3 from start 50: level-k pick = round(50*(2/3)^k)
    assert gen_level_k_problem(seed=0, depth=1)["answer"] in (27, 33, 40)  # depends on start
    a1 = gen_level_k_problem(seed=0, depth=1)["answer"]
    a3 = gen_level_k_problem(seed=0, depth=3)["answer"]
    assert a3 < a1                     # deeper level -> smaller number
