"""Free-text game-theory reasoning problems with exact verifiers.

Structure-first: sample an exact game, solve it exactly, render a faithful prose
problem (all quantities stated), and expose an exact `<answer>`-tag verifier.
Pure stdlib — must import under Python 3.9 (no torch/tinker).

Families (distinct reasoning operations, depth-parameterized):
  * bargaining        — finite alternating-offer SPE (backward induction). depth = rounds.
  * level_k           — p-beauty level-k pick (nested belief).            depth = k.
  * iterated_dominance — discrete Cournot solved by IESDS (deduction).    depth indexes size.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction

from game_theory_llm.reasoning.op_taxonomy import FAMILY_TAGS

_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def _rng(*key) -> random.Random:
    return random.Random(hash(key) & 0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# 1. Finite alternating-offer bargaining — backward induction (depth = rounds)
# --------------------------------------------------------------------------- #
def bargaining(seed: int, depth: int) -> dict:
    rng = _rng("barg", seed, depth)
    pie = 100
    delta = rng.choice([Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5)])
    T = depth
    share = Fraction(pie)                      # x_1: last proposer takes whole pie
    for _ in range(T - 1):
        share = Fraction(pie) - delta * share  # x_n = pie - delta * x_{n-1}
    answer = int(round(float(share)))
    a, b = rng.choice([("Ava", "Ben"), ("the buyer", "the seller"), ("Firm X", "Firm Y")])
    prompt = (
        f"{a} and {b} bargain over ${pie} by alternating offers for at most {T} round(s); "
        f"{a} makes the first offer. A rejected offer keeps the pie at ${pie}, but each side "
        f"values money one round later at a discount factor of {delta.numerator}/{delta.denominator} "
        f"(so $1 next round is worth ${delta.numerator}/{delta.denominator} this round). If the "
        f"final round {T} is reached with no deal, that round's proposer can take the entire ${pie}. "
        f"Both play the subgame-perfect equilibrium. Reason step by step using backward induction "
        f"from the last round, then give {a}'s equilibrium share, to the nearest whole dollar, as "
        f"<answer>DOLLARS</answer>."
    )
    return {"story_id": f"barg_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "bargaining", "framing": a, "op_tags": FAMILY_TAGS["bargaining"]}


# --------------------------------------------------------------------------- #
# 2. Level-k beauty contest — nested belief (depth = k)
# --------------------------------------------------------------------------- #
def level_k(seed: int, depth: int) -> dict:
    rng = _rng("lk", seed, depth)
    start = rng.choice([40, 50, 60])
    num, den = rng.choice([(2, 3), (1, 2), (3, 4)])
    val = float(start)
    for _ in range(depth):
        val = val * num / den
    answer = int(round(val))
    prompt = (
        f"Many players each pick a number in [0, 100]; the winner is closest to {num}/{den} of "
        f"the average of all picks. A level-0 player picks {start}. A level-k player assumes "
        f"everyone else is level-(k-1) and best-responds (picks {num}/{den} of what the level-(k-1) "
        f"players would pick). Reasoning exactly to level-{depth}, what number should a level-{depth} "
        f"player pick? Reason step by step, then answer with the nearest whole number as "
        f"<answer>NUMBER</answer>."
    )
    return {"story_id": f"lk_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "level_k", "framing": "contest", "op_tags": FAMILY_TAGS["level_k"]}


# --------------------------------------------------------------------------- #
# 3. Discrete Cournot duopoly — iterated elimination of dominated strategies
#    Symmetric Cournot-Nash q* = (a - c) / 3 (the unique IESDS survivor).
# --------------------------------------------------------------------------- #
def iterated_dominance(seed: int, depth: int) -> dict:
    rng = _rng("ied", seed, depth)
    c = rng.choice([0, 2, 4, 6])
    m = rng.randint(depth + 1, depth + 8)      # depth weakly indexes magnitude
    a = c + 3 * m                              # ensures q* = (a-c)/3 = m is integer
    answer = m
    f1, f2 = rng.choice([("Firm A", "Firm B"), ("Acme", "Beta"), ("two factories", "")][: 2])
    prompt = (
        f"Two firms ({f1} and {f2 or 'a rival'}) simultaneously choose how many units to produce. "
        f"The market price is {a} minus the total quantity produced by both firms, and each unit "
        f"costs {c} to make, so a firm's profit is its quantity times (price minus {c}). Each firm "
        f"maximizes its own profit and this is common knowledge. By iterated elimination of "
        f"dominated quantities, the game has a unique rational outcome. Reason step by step, then "
        f"give the quantity each firm produces as <answer>QUANTITY</answer>."
    )
    return {"story_id": f"ied_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "iterated_dominance", "framing": f1, "op_tags": FAMILY_TAGS["iterated_dominance"]}


# --------------------------------------------------------------------------- #
# 4. Subtraction game — combinatorial / modular reasoning (last stone wins)
#    P-positions: n % (K+1) == 0. Unique winning first move = n % (K+1).
# --------------------------------------------------------------------------- #
def subtraction_game(seed: int, depth: int) -> dict:
    rng = _rng("sub", seed, depth)
    K = rng.choice([3, 4, 5])
    n = rng.randint(depth * (K + 2), depth * (K + 2) + 3 * (K + 1))
    while n % (K + 1) == 0:                     # ensure a winning move exists
        n += 1
    answer = n % (K + 1)                        # unique optimal first removal
    a, b = rng.choice([("You", "a rival"), ("Ada", "Bo")])
    prompt = (
        f"There is a single pile of {n} stones. {a} and {b} alternate turns; on each turn a "
        f"player removes between 1 and {K} stones. Whoever removes the LAST stone wins. {a} "
        f"move(s) first and both play optimally. Reason step by step (work out which positions "
        f"are losing), then state how many stones {a} should remove on the first move as "
        f"<answer>NUMBER</answer>."
    )
    return {"story_id": f"sub_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "subtraction_game", "framing": a, "op_tags": FAMILY_TAGS["subtraction_game"]}


# --------------------------------------------------------------------------- #
# 5. Pure-strategy Nash equilibrium in an N x N bimatrix — constraint satisfaction
# --------------------------------------------------------------------------- #
def nash_pure(seed: int, depth: int) -> dict:
    rng = _rng("nash", seed, depth)
    N = min(depth + 1, 4)
    for _ in range(200):
        R = [[rng.randint(0, 9) for _ in range(N)] for _ in range(N)]   # row payoffs
        C = [[rng.randint(0, 9) for _ in range(N)] for _ in range(N)]   # col payoffs
        eq = [(i, j) for i in range(N) for j in range(N)
              if R[i][j] == max(R[r][j] for r in range(N))
              and C[i][j] == max(C[i][c] for c in range(N))]
        if len(eq) == 1:
            break
    else:
        return nash_pure(seed + 100000, depth)
    i, j = eq[0]
    answer = R[i][j]
    rows = "; ".join(
        "if Row plays {} and Column plays {}, payoffs are (Row {}, Column {})".format(
            "RTUV"[r], "rtuv"[c], R[r][c], C[r][c])
        for r in range(N) for c in range(N))
    prompt = (
        f"A one-shot game: Row chooses among {', '.join('RTUV'[:N])}; Column chooses among "
        f"{', '.join('rtuv'[:N])}. {rows}. A pure-strategy Nash equilibrium is a pair where "
        f"neither player can do better by unilaterally switching. Reason step by step to find "
        f"the unique pure Nash equilibrium, then give Row's payoff there as <answer>NUMBER</answer>."
    )
    return {"story_id": f"nash_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "nash_pure", "op_tags": FAMILY_TAGS["nash_pure"]}


# --------------------------------------------------------------------------- #
# 6. Second-price auction optimal bid — counterfactual / expected value
# --------------------------------------------------------------------------- #
def second_price_auction(seed: int, depth: int) -> dict:
    rng = _rng("auc", seed, depth)
    n = depth + 1
    v = rng.randint(20, 80)
    others = [rng.randint(5, 95) for _ in range(n)]
    hi = max(others)
    answer = max(0, v - hi) if v > hi else 0     # win iff v>hi; pay 2nd price = hi
    prompt = (
        f"A second-price sealed-bid auction: the highest bidder wins and pays the "
        f"second-highest bid. Your private value for the item is ${v}. The other bidders bid "
        f"their values: {others}. Bidding optimally (truthfully), what is your profit "
        f"(your value minus the price you pay, or 0 if you do not win)? Reason step by step, "
        f"then answer <answer>DOLLARS</answer>."
    )
    return {"story_id": f"auc_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "second_price_auction",
            "op_tags": FAMILY_TAGS["second_price_auction"]}


# --------------------------------------------------------------------------- #
# 7. Three-player Shapley value — combinatorial aggregation
# --------------------------------------------------------------------------- #
def shapley3(seed: int, depth: int) -> dict:
    rng = _rng("shap", seed, depth)
    for _ in range(400):
        v = {frozenset(s): rng.randint(0, 18) for s in
             [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]}
        v[frozenset()] = 0
        # phi_1 = (1/6)[2(v1) + (v12 - v2) + (v13 - v3) + 2(v123 - v23)]
        num = (2 * v[frozenset((1,))] + (v[frozenset((1, 2))] - v[frozenset((2,))])
               + (v[frozenset((1, 3))] - v[frozenset((3,))])
               + 2 * (v[frozenset((1, 2, 3))] - v[frozenset((2, 3))]))
        if num % 6 == 0:
            break
    else:
        return shapley3(seed + 100000, depth)
    answer = num // 6
    g = lambda *s: v[frozenset(s)]
    prompt = (
        f"Three players (1, 2, 3) can form coalitions worth: {{1}}={g(1)}, {{2}}={g(2)}, "
        f"{{3}}={g(3)}, {{1,2}}={g(1,2)}, {{1,3}}={g(1,3)}, {{2,3}}={g(2,3)}, "
        f"{{1,2,3}}={g(1,2,3)} (empty coalition = 0). The Shapley value of a player is their "
        f"average marginal contribution over all 6 orderings of the players. Reason step by "
        f"step, then give player 1's Shapley value as <answer>NUMBER</answer>."
    )
    return {"story_id": f"shap_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "shapley3", "op_tags": FAMILY_TAGS["shapley3"]}


# --------------------------------------------------------------------------- #
# 8. Prose minimax game tree — backward induction over a nested tree
# --------------------------------------------------------------------------- #
def minimax_prose(seed: int, depth: int) -> dict:
    from game_theory_llm.reasoning import gametree
    p = gametree.gen_game_tree_problem(seed, depth, branching=2, framing="negotiation")
    return {"story_id": f"mmx_d{depth}_{seed}", "prompt": p["prompt"], "answer": p["answer"],
            "depth": depth, "family": "minimax_prose", "op_tags": FAMILY_TAGS["minimax_prose"]}


FAMILIES = {"bargaining": bargaining, "level_k": level_k,
            "iterated_dominance": iterated_dominance,
            "subtraction_game": subtraction_game,
            "nash_pure": nash_pure, "second_price_auction": second_price_auction,
            "shapley3": shapley3, "minimax_prose": minimax_prose}


def extract_answer(text: str):
    """Integer answer from an <answer>N</answer> tag, else None."""
    m = _ANS.search(text or "")
    return int(m.group(1)) if m else None


def verify(response: str, problem: dict) -> bool:
    pred = extract_answer(response)
    return pred is not None and pred == problem["answer"]
