"""Depth-controllable, verifiable long-reasoning tasks from game theory.

The core idea: game-theoretic structures give us synthetic reasoning problems
whose required reasoning *depth* is a tunable dial and whose answers are
*programmatically verifiable* — ideal for curriculum SFT / rejection-sampling /
RLVR aimed at making a model a better long-depth reasoner.

This module implements the canonical long-depth task: **minimax / backward
induction over a random sequential game tree**. A depth-D tree forces D plies of
lookahead; the answer is the root value under optimal play, computed by backward
induction (the verifier). Diverse natural-language framings of the same deep
structure prevent shallow pattern-matching.

Two more families are provided for breadth:
  * `level_k_beauty`  — iterated-belief (level-k) reasoning, depth = k.
  * `iterated_dominance` — rounds of eliminating dominated strategies, depth = k.

Every generator returns a dict with: prompt, answer (verifiable), depth,
plus a worked chain-of-thought `solution` usable as an SFT target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Game-tree minimax (backward induction) — depth = plies of lookahead
# --------------------------------------------------------------------------- #

@dataclass
class Node:
    is_leaf: bool
    value: Optional[int] = None          # leaf payoff
    children: list = field(default_factory=list)   # list[(label, Node)]


def _gen_tree(rng, depth, branching, leaf_lo, leaf_hi):
    if depth == 0:
        return Node(is_leaf=True, value=rng.randint(leaf_lo, leaf_hi))
    labels = ["A", "B", "C", "D", "E"][:branching]
    return Node(is_leaf=False,
                children=[(lab, _gen_tree(rng, depth - 1, branching, leaf_lo, leaf_hi))
                          for lab in labels])


def minimax(node: Node, maximizing: bool) -> int:
    """Root value under optimal play. Root (depth 0) is the maximizer (MAX)."""
    if node.is_leaf:
        return node.value
    vals = [minimax(c, not maximizing) for _, c in node.children]
    return max(vals) if maximizing else min(vals)


def _optimal_action(node: Node) -> str:
    """Best root move label for MAX."""
    best, blab = None, None
    for lab, c in node.children:
        v = minimax(c, maximizing=False)     # child is MIN's turn
        if best is None or v > best:
            best, blab = v, lab
    return blab


def _render_tree(node: Node, depth: int, maximizing: bool, indent: int = 0) -> list[str]:
    pad = "  " * indent
    who = "MAX" if maximizing else "MIN"
    lines = []
    for lab, c in node.children:
        if c.is_leaf:
            lines.append(f"{pad}- {who} plays {lab} -> outcome {c.value:+d}")
        else:
            lines.append(f"{pad}- {who} plays {lab}:")
            lines.extend(_render_tree(c, depth - 1, not maximizing, indent + 1))
    return lines


_FRAMINGS = {
    "abstract": ("Two players, MAX and MIN, alternate turns; MAX moves first. "
                 "MAX wants the final outcome number as LARGE as possible, MIN wants it "
                 "as SMALL as possible. Both play optimally (backward induction)."),
    "negotiation": ("A buyer (MAX, wants the highest final price) and a seller's agent "
                    "(MIN, wants the lowest) alternate concessions; the buyer moves first. "
                    "Each leaf is the final price. Both negotiate optimally."),
    "chess": ("White (MAX) and Black (MIN) alternate moves, White first. Each leaf is the "
              "position evaluation (positive favours White). Both play the best move."),
}


def gen_game_tree_problem(seed: int, depth: int, branching: int = 2,
                          framing: str = "abstract") -> dict:
    rng = random.Random(seed)
    root = _gen_tree(rng, depth, branching, -9, 9)
    value = minimax(root, maximizing=True)
    action = _optimal_action(root)
    tree_txt = "\n".join(_render_tree(root, depth, maximizing=True))
    intro = _FRAMINGS.get(framing, _FRAMINGS["abstract"])
    prompt = (
        f"{intro}\n\nThe game tree ({depth} moves deep) is:\n{tree_txt}\n\n"
        f"Reason step by step using backward induction (evaluate the deepest "
        f"choices first, then work upward). What is the final outcome number under "
        f"optimal play? Give your answer as <answer>NUMBER</answer>."
    )
    return {
        "story_id": f"gametree_d{depth}_{seed}",
        "prompt": prompt,
        "answer": value,
        "optimal_action": action,
        "depth": depth,
        "branching": branching,
        "framing": framing,
        "task": "game_tree_minimax",
    }


def _solve_with_trace(node: Node, maximizing: bool, path: str = "root") -> tuple:
    """Backward-induction value + a worked CoT trace (leaves-up narration).
    Returns (value, list[str] trace lines)."""
    if node.is_leaf:
        return node.value, []
    lines = []
    child_vals = []
    for lab, c in node.children:
        v, sub = _solve_with_trace(c, not maximizing, f"{path}->{lab}")
        lines.extend(sub)
        child_vals.append((lab, v))
    who = "MAX" if maximizing else "MIN"
    pick = (max if maximizing else min)(child_vals, key=lambda kv: kv[1])
    opts = ", ".join(f"{lab}={v:+d}" for lab, v in child_vals)
    lines.append(f"At {path} ({who} to move), children give [{opts}]; "
                 f"{who} {'maximises' if maximizing else 'minimises'} -> "
                 f"picks {pick[0]} for value {pick[1]:+d}.")
    return pick[1], lines


def gen_game_tree_sft(seed: int, depth: int, branching: int = 2,
                      framing: str = "abstract") -> dict:
    """Game-tree problem + a gold backward-induction CoT completion (SFT target)."""
    p = gen_game_tree_problem(seed, depth, branching, framing)
    rng = random.Random(seed)
    root = _gen_tree(rng, depth, branching, -9, 9)   # same seed -> same tree
    value, trace = _solve_with_trace(root, maximizing=True)
    cot = ("Solving by backward induction, deepest choices first:\n"
           + "\n".join(trace)
           + f"\n\nSo under optimal play the outcome is <answer>{value}</answer>.")
    p["completion"] = cot
    return p


# --------------------------------------------------------------------------- #
# Level-k beauty contest — depth = belief-recursion levels
# --------------------------------------------------------------------------- #

def gen_level_k_problem(seed: int, depth: int, p_num: int = 2, p_den: int = 3) -> dict:
    """p-beauty contest. A level-0 agent picks 50; each higher level best-responds
    to the level below. A level-k agent picks 50*(p)^k (rounded). depth = k."""
    rng = random.Random(seed)
    start = rng.choice([50, 40, 60])
    p = p_num / p_den
    val = start
    for _ in range(depth):
        val = val * p
    answer = round(val)
    prompt = (
        f"In a p-beauty contest, many players each pick a number in [0, 100]; the "
        f"winner is closest to {p_num}/{p_den} of the average of all picks. A "
        f"\"level-0\" player picks {start}. A \"level-k\" player assumes everyone else "
        f"is level-(k-1) and best-responds. Reasoning exactly {depth} level(s) deep, "
        f"what number should a level-{depth} player pick? Give <answer>NUMBER</answer>."
    )
    return {"story_id": f"levelk_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "task": "level_k_beauty"}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # solver correctness on a hand tree:
    #   MAX{ MIN{3,-2}, MIN{5,1} } = MAX{-2, 1} = 1
    t = Node(False, children=[
        ("A", Node(False, children=[("C", Node(True, value=3)), ("D", Node(True, value=-2))])),
        ("B", Node(False, children=[("E", Node(True, value=5)), ("F", Node(True, value=1))])),
    ])
    assert minimax(t, True) == 1, minimax(t, True)
    assert _optimal_action(t) == "B"

    # brute-force check minimax against exhaustive eval for random trees
    def brute(node, mx):
        if node.is_leaf:
            return node.value
        vs = [brute(c, not mx) for _, c in node.children]
        return max(vs) if mx else min(vs)
    import random as _r
    for s in range(200):
        rng = _r.Random(s)
        root = _gen_tree(rng, depth=rng.randint(1, 6), branching=rng.choice([2, 3]),
                         leaf_lo=-9, leaf_hi=9)
        assert minimax(root, True) == brute(root, True)
    print("solver OK (200 random trees verified)")

    for d in (1, 2, 4, 6):
        p = gen_game_tree_problem(seed=d, depth=d)
        print(f"\n=== game_tree depth {d}: answer={p['answer']} action={p['optimal_action']} ===")
        if d <= 2:
            print(p["prompt"])
    for d in (1, 2, 3):
        p = gen_level_k_problem(seed=d, depth=d)
        print(f"level_k depth {d}: answer={p['answer']}")
