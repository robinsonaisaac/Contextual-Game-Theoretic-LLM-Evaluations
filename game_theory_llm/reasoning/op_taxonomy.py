"""Reasoning-operation taxonomy: the analytical backbone of the transfer experiment.

Every training family and every eval item is labelled with one or more operation tags.
The operation-overlap statistic (does an eval item improve in proportion to how much its
operation was trained?) distinguishes operation-specific transfer (H_op) from general
reasoning discipline (H_gen). Pure stdlib.
"""
from __future__ import annotations

OP_TAGS = {
    "backward_induction",      # multi-step lookahead / planning from the end
    "iterated_elimination",    # deduction by removing dominated/invalid options
    "nested_belief",           # reasoning about others' reasoning (k-level / ToM)
    "modular_combinatorial",   # parity/XOR/mod invariants, counting
    "constraint_satisfaction", # find a consistent joint assignment
    "expected_value",          # probabilistic / counterfactual value
    "recursion_nesting",       # nested structure (brackets, proofs, state over steps)
    "arithmetic_search",       # search numeric combinations to a target
}

# Training families -> operation tags
FAMILY_TAGS = {
    "bargaining": ["backward_induction"],
    "level_k": ["nested_belief"],
    "iterated_dominance": ["iterated_elimination"],
    "subtraction_game": ["modular_combinatorial"],
    # new (Tier-1 eval-only; Tier-2 trainable)
    "nash_pure": ["constraint_satisfaction"],
    "second_price_auction": ["expected_value"],
    "shapley3": ["modular_combinatorial", "expected_value"],
    "minimax_prose": ["backward_induction", "recursion_nesting"],
}

# Eval benchmarks -> operation tags + knowledge flag
BENCH_TAGS = {
    "dyck": ["recursion_nesting"],
    "prontoqa": ["iterated_elimination"],
    "countdown": ["arithmetic_search"],
    "ordering": ["constraint_satisfaction"],
    "knights_knaves": ["iterated_elimination", "nested_belief"],
    "boolean_eval": ["recursion_nesting"],
    "musr": ["backward_induction", "constraint_satisfaction"],
    "gsm_symbolic": ["arithmetic_search", "backward_induction"],
    "mmlu_pro": ["constraint_satisfaction"],   # mixed; knowledge-heavy
    "gpqa": ["constraint_satisfaction"],       # mixed; knowledge-heavy
}
BENCH_KNOWLEDGE = {b: (b in {"mmlu_pro", "gpqa"}) for b in BENCH_TAGS}

# The pre-registered TRAINED operations for the Tier-1 overlap experiment
TRAINED_TAGS = {"backward_induction", "nested_belief",
                "iterated_elimination", "modular_combinatorial"}


def overlap(item_tags, trained_tags=TRAINED_TAGS) -> float:
    """Fraction of an item's required operations that were trained."""
    it = set(item_tags)
    if not it:
        return 0.0
    return len(it & set(trained_tags)) / len(it)
