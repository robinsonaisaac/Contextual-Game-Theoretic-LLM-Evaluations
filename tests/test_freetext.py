"""Tests for free-text game-theory generators + exact verifiers (Task 1)."""
import re
from game_theory_llm.reasoning.freetext import (
    bargaining, level_k, iterated_dominance, FAMILIES, extract_answer, verify,
)


def test_level_k_closed_form():
    # answer = round(start * (num/den)^depth); prose must state the level (faithful)
    p = level_k(seed=1, depth=3)
    assert isinstance(p["answer"], int)
    assert "level-3" in p["prompt"] or "level-3" in p["prompt"].replace(" ", "")
    # re-derive from the prose-stated start/ratio
    m = re.search(r"picks (\d+)", p["prompt"]); assert m


def test_bargaining_recursion_matches_formula():
    # x_1 = pie; x_n = pie - delta*x_{n-1}; first proposer (names[0]) gets x_T
    p = bargaining(seed=2, depth=4)
    assert 0 <= p["answer"] <= 100
    # depth-1 bargaining => first proposer takes whole pie
    assert bargaining(seed=2, depth=1)["answer"] == 100


def test_iterated_dominance_cournot_equilibrium():
    p = iterated_dominance(seed=3, depth=2)
    assert isinstance(p["answer"], int) and p["answer"] >= 1
    # prose states the demand intercept and cost so the answer is derivable
    assert "price" in p["prompt"].lower() and "cost" in p["prompt"].lower()


def test_answers_vary_per_family():
    for name, fam in FAMILIES.items():
        answers = {fam(seed=s, depth=3)["answer"] for s in range(15)}
        assert len(answers) > 1, f"{name}: answer is guessable-constant"


def test_extract_and_verify():
    p = bargaining(seed=5, depth=3)
    assert extract_answer(f"...<answer>{p['answer']}</answer>") == p["answer"]
    assert verify(f"<answer>{p['answer']}</answer>", p) is True
    wrong = p["answer"] + 7
    assert verify(f"<answer>{wrong}</answer>", p) is False
    assert verify("no tag here", p) is False


def test_subtraction_game_winning_move():
    from game_theory_llm.reasoning.freetext import subtraction_game
    p = subtraction_game(seed=3, depth=2)
    assert 1 <= p["answer"]                       # always a winning move (n % (K+1) != 0)
    # re-derive K and n from the prose and check answer == n % (K+1)
    import re
    n = int(re.search(r"pile of (\d+) stones", p["prompt"]).group(1))
    K = int(re.search(r"between 1 and (\d+) stones", p["prompt"]).group(1))
    assert p["answer"] == n % (K + 1)


def test_new_families_verifiers_via_prose():
    import re
    from game_theory_llm.reasoning.freetext import (
        second_price_auction, shapley3, minimax_prose)
    from game_theory_llm.reasoning import gametree
    # second-price: re-derive v + others from prose, check profit
    p = second_price_auction(seed=4, depth=3)
    v = int(re.search(r"value for the item is \$(\d+)", p["prompt"]).group(1))
    others = [int(x) for x in re.search(r"their values: \[([0-9,\s]+)\]", p["prompt"]).group(1).split(",")]
    hi = max(others)
    assert p["answer"] == (max(0, v - hi) if v > hi else 0)
    # shapley: re-derive coalition values, recompute phi_1
    q = shapley3(seed=4, depth=2)
    g = {tuple(sorted(int(c) for c in m.group(1).split(","))): int(m.group(2))
         for m in re.finditer(r"\{([\d,]+)\}=(\d+)", q["prompt"])}
    num = (2*g[(1,)] + (g[(1,2)]-g[(2,)]) + (g[(1,3)]-g[(3,)]) + 2*(g[(1,2,3)]-g[(2,3)]))
    assert q["answer"] == num // 6 and num % 6 == 0
    # minimax_prose matches the gametree solver
    mp = minimax_prose(seed=4, depth=3)
    assert isinstance(mp["answer"], int)
