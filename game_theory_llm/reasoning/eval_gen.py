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

_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}


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
    return {"story_id": f"dyck_d{depth}_{seed}", "prompt": prompt, "answer": closers,
            "depth": depth, "family": "dyck"}


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
    return {"story_id": f"pqa_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "prontoqa"}
