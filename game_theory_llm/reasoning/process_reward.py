"""Process reward for game-tree backward induction — exact step labels from the solver.

The dissociation hypothesis from Tier-1b/2: outcome-only GRPO teaches depth-specific
shortcuts (in-domain gains, zero extrapolation). A correct backward-induction CoT must
compute every internal node's minimax value bottom-up; the solver gives those values
exactly, so we can reward the *procedure* instead of just the final integer.

Score = order-aware recall of the gold internal-node value sequence (post-order, the
same order `gametree._solve_with_trace` narrates), via longest-common-subsequence
between the gold sequence and the integers appearing in the response, with a spray
guard: emitting far more integers than honest reasoning needs is penalized, so
cycling -9..9 to "hit" values caps well below an honest trace's score.

Pure stdlib; must import under Python 3.9 (used by the corpus builder) and inside the
Tinker 3.11 venv (used by the RL env, which inlines nothing — it imports this file's
logic via copy in scripts/gt_rl_env.py... no: scripts add this path; keep stdlib-only).
"""
from __future__ import annotations

import random
import re

from game_theory_llm.reasoning.gametree import _gen_tree, minimax

_INT = re.compile(r"-?\d+")
SPRAY_FACTOR = 8          # honest traces use <= ~5x gold-length integers; penalize beyond 8x
MAX_RESP_INTS = 4000      # hard cap for O(G*N) LCS cost


def gold_process_values(seed: int, depth: int, branching: int = 2) -> list:
    """Internal-node minimax values in post-order (deepest-first, left-to-right) —
    exactly the order gametree._solve_with_trace narrates them. Root value is last."""
    rng = random.Random(seed)
    root = _gen_tree(rng, depth, branching, -9, 9)

    out = []

    def post(node, maximizing):
        if node.is_leaf:
            return
        for _, c in node.children:
            post(c, not maximizing)
        out.append(minimax(node, maximizing))

    post(root, True)
    return out


def extract_ints(text: str) -> list:
    return [int(m) for m in _INT.findall(text or "")][:MAX_RESP_INTS]


def lcs_len(a: list, b: list) -> int:
    """Length of the longest common subsequence (order-aware, gaps allowed)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def process_score(response: str, gold: list) -> float:
    """Order-aware recall of the gold value sequence, spray-guarded, in [0, 1]."""
    if not gold:
        return 0.0
    ints = extract_ints(response)
    if not ints:
        return 0.0
    recall = lcs_len(gold, ints) / len(gold)
    spray_penalty = min(1.0, (SPRAY_FACTOR * len(gold)) / len(ints))
    return recall * spray_penalty


def combined_reward(response: str, gold: list, outcome_correct: bool,
                    w_outcome: float = 0.5) -> float:
    """Process arm's graded reward: w*outcome + (1-w)*process, in [0, 1]."""
    return w_outcome * float(outcome_correct) + (1.0 - w_outcome) * process_score(response, gold)
