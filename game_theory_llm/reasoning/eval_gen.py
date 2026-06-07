"""Procedurally-generated, depth-scaled, verifiable NON-game reasoning evals.

Used only for evaluation (transfer diagnostics) — these are NOT game-theoretic and
NOT in the training distribution, so improvement on them is genuine transfer.
Pure stdlib.

  * dyck     — close a sequence of opened brackets in the right order.
               nesting depth = the structural analog of game-tree depth.
  * prontoqa — synthetic multi-hop entailment ('every X is Y' chains).
               depth = number of proof hops.
"""
from __future__ import annotations

import random

from game_theory_llm.reasoning.op_taxonomy import BENCH_TAGS, BENCH_KNOWLEDGE

_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _tag(d: dict) -> dict:
    fam = d["family"]
    d["op_tags"] = BENCH_TAGS.get(fam, [])
    d["knowledge"] = BENCH_KNOWLEDGE.get(fam, False)
    return d


def _rng(*key):
    return random.Random(hash(key) & 0x7FFFFFFF)


def dyck(seed: int, depth: int) -> dict:
    rng = _rng("dyck", seed, depth)
    opens = [rng.choice(list(_PAIRS)) for _ in range(depth)]
    closers = "".join(_PAIRS[c] for c in reversed(opens))
    prompt = (
        f"You are given an opening bracket sequence. Output ONLY the bracket(s) needed to "
        f"close it correctly, in the right order.\nInput: {''.join(opens)}\n"
        f"Reason step by step, then give the closing brackets as <answer>CLOSERS</answer>."
    )
    return _tag({"story_id": f"dyck_d{depth}_{seed}", "prompt": prompt, "answer": closers,
                 "depth": depth, "family": "dyck"})


def prontoqa(seed: int, depth: int) -> dict:
    """A chain of `depth` rules 'Every A_i is an A_{i+1}.' plus 'Max is an A_0.'
    Half the time the queried predicate is the chain endpoint (entailed=True);
    otherwise an unrelated predicate (False). Rules are shuffled in the prompt."""
    rng = _rng("pqa", seed, depth)
    chain = [f"item{i}" for i in range(depth + 1)]
    rules = [f"Every {chain[i]} is a {chain[i + 1]}." for i in range(depth)]
    shown = rules[:]
    rng.shuffle(shown)
    subject = "Max"
    fact = f"{subject} is a {chain[0]}."
    if rng.random() < 0.5:
        target, answer = chain[depth], "True"      # follows by the full chain
    else:
        target, answer = f"thing{seed % 97}", "False"   # unrelated -> not entailed
    body = " ".join(shown) + " " + fact
    prompt = (
        f"{body}\nQuestion: Does it follow that '{subject} is a {target}.'? Reason step by "
        f"step through the rules, then answer <answer>True</answer> or <answer>False</answer>."
    )
    return _tag({"story_id": f"pqa_d{depth}_{seed}", "prompt": prompt, "answer": answer,
                 "depth": depth, "family": "prontoqa"})


def countdown(seed: int, depth: int) -> dict:
    """Insert +,-,* between `depth` numbers (in order, evaluated strictly left-to-right)
    to MAXIMIZE the result. answer = the maximum. tag=arithmetic_search."""
    import itertools
    rng = _rng("cd", seed, depth)
    nums = [rng.randint(1, 9) for _ in range(depth + 1)]
    best = None
    for ops in itertools.product("+-*", repeat=len(nums) - 1):
        val = nums[0]
        for op, x in zip(ops, nums[1:]):
            val = val + x if op == "+" else val - x if op == "-" else val * x
        best = val if best is None else max(best, val)
    prompt = (
        f"Insert one of +, -, or * between each pair of these numbers, in the given order, and "
        f"evaluate STRICTLY LEFT TO RIGHT (ignore normal operator precedence), to make the value "
        f"as large as possible. Numbers: {nums}. Reason step by step, then give the maximum value "
        f"as <answer>NUMBER</answer>."
    )
    return _tag({"story_id": f"cd_d{depth}_{seed}", "prompt": prompt, "answer": best,
                 "depth": depth, "family": "countdown"})


def ordering(seed: int, depth: int) -> dict:
    """`depth` items in a hidden total order; give all consecutive 'X is before Y' clues
    (shuffled) -> unique order; ask the position (1=first) of a queried item.
    tag=constraint_satisfaction."""
    rng = _rng("ord", seed, depth)
    items = [f"item{c}" for c in "ABCDEFG"[:depth + 1]]
    order = items[:]
    rng.shuffle(order)
    clues = [f"{order[i]} is somewhere before {order[i + 1]}" for i in range(len(order) - 1)]
    rng.shuffle(clues)
    q = rng.choice(items)
    answer = order.index(q) + 1
    prompt = (
        f"There are {len(items)} items: {', '.join(items)}. They are arranged in a single line "
        f"(position 1 = first). Clues: {'; '.join(clues)}. Reason step by step to the unique "
        f"ordering, then give the position (a number) of {q} as <answer>NUMBER</answer>."
    )
    return _tag({"story_id": f"ord_d{depth}_{seed}", "prompt": prompt, "answer": answer,
                 "depth": depth, "family": "ordering"})


def knights_knaves(seed: int, depth: int) -> dict:
    """`depth`+1 islanders (knight=truth, knave=lie). A narrator-GIVEN fact anchors P1's type
    (breaking the all-flip symmetry that type-only statements always carry); each other
    islander states the type of the previous one (connected chain) -> UNIQUE solution.
    answer = number of knights. tag=iterated_elimination+nested_belief."""
    rng = _rng("kk", seed, depth)
    n = depth + 1
    names = [f"P{i+1}" for i in range(n)]
    types = [rng.random() < 0.5 for _ in range(n)]
    types[0] = True                                   # GIVEN: P1 is a knight (external anchor)
    rel = []                                          # (i, j, claim_is_knight)
    lines = []
    for i in range(1, n):
        j = i - 1
        claim = types[j] if types[i] else (not types[j])     # truth of claim == types[i]
        rel.append((i, j, claim))
        lines.append(f'{names[i]} says "{names[j]} is a {"knight" if claim else "knave"}"')

    def consistent(asg):
        if not asg[0]:                                # the given fact: P1 is a knight
            return False
        for (i, j, claim) in rel:
            if asg[i] != (asg[j] == claim):
                return False
        return True
    assert sum(1 for a in _bits(n) if consistent(a)) == 1     # unique by construction
    answer = sum(types)
    prompt = (
        f"On an island, knights always tell the truth and knaves always lie. It is given that "
        f"{names[0]} is a knight. {'; '.join(lines)}. Reason step by step to the unique "
        f"consistent assignment, then give the total number of knights as <answer>NUMBER</answer>."
    )
    return _tag({"story_id": f"kk_d{depth}_{seed}", "prompt": prompt, "answer": answer,
                 "depth": depth, "family": "knights_knaves"})


def _bits(n):
    for x in range(2 ** n):
        yield [bool((x >> k) & 1) for k in range(n)]


def boolean_eval(seed: int, depth: int) -> dict:
    """Nested boolean expression of nesting `depth`; answer = 1/0. tag=recursion_nesting."""
    rng = _rng("bool", seed, depth)

    def build(d):
        if d == 0:
            v = rng.random() < 0.5
            return ("True" if v else "False"), v
        op = rng.choice(["and", "or", "not"])
        if op == "not":
            s, v = build(d - 1)
            return f"(not {s})", (not v)
        ls, lv = build(d - 1)
        rs, rv = build(d - 1)
        return (f"({ls} {op} {rs})", (lv and rv) if op == "and" else (lv or rv))

    expr, val = build(depth)
    prompt = (
        f"Evaluate this boolean expression (and/or/not over True/False): {expr}\n"
        f"Reason step by step from the innermost parentheses outward, then answer 1 for True or "
        f"0 for False as <answer>NUMBER</answer>."
    )
    return _tag({"story_id": f"bool_d{depth}_{seed}", "prompt": prompt, "answer": 1 if val else 0,
                 "depth": depth, "family": "boolean_eval"})
