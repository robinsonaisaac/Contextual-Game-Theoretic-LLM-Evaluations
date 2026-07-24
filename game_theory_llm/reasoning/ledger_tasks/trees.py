"""Minimax game-tree family. Wraps gametree._gen_tree/minimax. Ledger content: each
internal node's backward-induction value keyed by its path id (`v.<path>`); deps = the
child node keys. horizon knob = depth d3-d7 chosen from the requested horizon."""
from __future__ import annotations

import math
import random
from typing import List

from game_theory_llm.reasoning.gametree import _gen_tree
from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10


def _depth_for_horizon(horizon: int) -> int:
    d = int(round(math.log2(max(2, horizon) + 1)))
    return max(3, min(7, d))


def _render_leaves(node, path, maximizing, lines):
    who = "MAX" if maximizing else "MIN"
    for lab, c in node.children:
        p = f"{path}{lab}"
        if c.is_leaf:
            lines.append(f"- {who} plays {lab} -> outcome {c.value:+d}  [{p}]")
        else:
            lines.append(f"- {who} plays {lab}:  [{p}]")
            _render_leaves(c, p, not maximizing, lines)


def _walk(node, path, maximizing, notes):
    if node.is_leaf:
        return node.value, f"leaf.{path}"
    child = []
    for lab, c in node.children:
        v, k = _walk(c, f"{path}{lab}", not maximizing, notes)
        child.append((v, k))
    val = (max if maximizing else min)(cv for cv, _ in child)
    key = f"v.{path}"
    notes.append(Note(key=key, value=str(val), deps=[k for _, k in child]))
    return val, key


def gen(seed: int, horizon: int) -> LedgerTask:
    depth = _depth_for_horizon(horizon)
    rng = random.Random(seed)
    root = _gen_tree(rng, depth, branching=2, leaf_lo=-9, leaf_hi=9)
    notes: List[Note] = []
    root_val, _ = _walk(root, "R", True, notes)
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    events.append(Answer(value=str(root_val)))
    gold_facts = {n.key: n.value for n in notes}
    lines: List[str] = []
    _render_leaves(root, "R", True, lines)
    prompt = (
        "Two players MAX and MIN alternate turns; MAX moves first and wants the final "
        "outcome as LARGE as possible, MIN as SMALL as possible; both play optimally.\n"
        f"Game tree ({depth} moves deep); each node is tagged with its path id in [brackets]:\n"
        + "\n".join(lines) + "\n\n"
        "Work bottom-up (backward induction). Keep an explicit running ledger of each "
        "internal node's value keyed by its path id, then read off the root value.\n"
        "End with a line: ANSWER: <the root value under optimal play>."
    )
    return LedgerTask(
        family="trees", task_id=f"trees_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=str(root_val), gold_facts=gold_facts, events=events,
        horizon=len(notes), needs_ledger=True,
    )
