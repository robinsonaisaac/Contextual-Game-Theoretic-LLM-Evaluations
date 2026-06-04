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
            "depth": depth, "family": "bargaining", "framing": a}


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
            "depth": depth, "family": "level_k", "framing": "contest"}


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
            "depth": depth, "family": "iterated_dominance", "framing": f1}


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
            "depth": depth, "family": "subtraction_game", "framing": a}


FAMILIES = {"bargaining": bargaining, "level_k": level_k,
            "iterated_dominance": iterated_dominance,
            "subtraction_game": subtraction_game}


def extract_answer(text: str):
    """Integer answer from an <answer>N</answer> tag, else None."""
    m = _ANS.search(text or "")
    return int(m.group(1)) if m else None


def verify(response: str, problem: dict) -> bool:
    pred = extract_answer(response)
    return pred is not None and pred == problem["answer"]
