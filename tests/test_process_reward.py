"""Validate the process reward: gold-trace agreement, honest-vs-spray separation."""
import pytest

from game_theory_llm.reasoning.gametree import gen_game_tree_sft, Node, minimax
from game_theory_llm.reasoning.process_reward import (
    gold_process_values, process_score, combined_reward, lcs_len, extract_ints,
)


def test_gold_values_hand_tree():
    # MAX{ MIN{3,-2}=-2, MIN{5,1}=1 } = 1 ; post-order: [-2, 1, 1]
    vals = gold_process_values(seed=0, depth=2)
    assert len(vals) == 3                      # 2^2 - 1 internal nodes
    # root value (last in post-order) must equal full minimax
    import random as _r
    from game_theory_llm.reasoning.gametree import _gen_tree
    root = _gen_tree(_r.Random(0), 2, 2, -9, 9)
    assert vals[-1] == minimax(root, True)


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_gold_cot_scores_near_one(depth):
    """The solver's own gold CoT must earn ~full process credit (in order, no spray)."""
    for seed in range(10):
        p = gen_game_tree_sft(seed=seed, depth=depth)
        gold = gold_process_values(seed=seed, depth=depth)
        assert len(gold) == 2 ** depth - 1
        assert gold[-1] == p["answer"]          # root value is the final answer
        s = process_score(p["completion"], gold)
        assert s >= 0.95, (depth, seed, s)


def test_spray_attack_capped():
    """Cycling the whole value range must score far below an honest trace."""
    gold = gold_process_values(seed=3, depth=5)          # 31 values
    spray = " ".join(str(v) for _ in range(len(gold)) for v in range(-9, 10))
    s = process_score(spray, gold)
    assert s <= 0.45, s                                   # spray ceiling ~ 8/19


def test_truncated_trace_partial_credit():
    """Half a correct trace earns roughly half credit (dense signal for GRPO)."""
    p = gen_game_tree_sft(seed=7, depth=4)
    gold = gold_process_values(seed=7, depth=4)
    lines = p["completion"].splitlines()
    half = "\n".join(lines[: len(lines) // 2])
    s_half, s_full = process_score(half, gold), process_score(p["completion"], gold)
    assert 0.2 <= s_half <= 0.8 and s_full >= 0.95


def test_combined_reward_weights():
    p = gen_game_tree_sft(seed=1, depth=3)
    gold = gold_process_values(seed=1, depth=3)
    assert combined_reward(p["completion"], gold, outcome_correct=True) >= 0.97
    assert combined_reward("", gold, outcome_correct=False) == 0.0
    # right answer, no work shown -> only the outcome half
    bare = f"<answer>{p['answer']}</answer>"
    assert 0.5 <= combined_reward(bare, gold, True) <= 0.62


def test_lcs_and_extract():
    assert lcs_len([1, 2, 3], [9, 1, 5, 2, 3]) == 3
    assert extract_ints("vals -2, +7 then 13") == [-2, 7, 13]
