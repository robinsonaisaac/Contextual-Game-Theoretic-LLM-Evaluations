"""Two-pass eval: pruning at +/- alpha_max, then full sweep on top-k cells."""

from __future__ import annotations

from typing import Any

import torch

from .application import steering_hook, generate_with_hook
from .extraction import parse_decision, is_cooperative
from .models import SteeringEvalResult, SteeringVectorSet


def _eval_one_cell(
    model,
    tokenizer,
    layers,
    vec,
    alpha: float,
    stories: list[dict[str, Any]],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> SteeringEvalResult:
    """Evaluate one (layer, position, alpha) cell across all eval stories."""
    decisions = []
    n_coop = 0
    with steering_hook(layers, vec, alpha):
        for s in stories:
            text = generate_with_hook(
                model, tokenizer, s["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=s.get("seed"),
            )
            d = parse_decision(text)
            cooperated = is_cooperative(d, s["coop_choice"])
            decisions.append({
                "story_id": s["story_id"],
                "decision": d,
                "cooperated": cooperated,
                "trace": text,
            })
            if cooperated:
                n_coop += 1
    return SteeringEvalResult(
        layer=vec.layer,
        position=vec.position,
        alpha=alpha,
        n_stories=len(stories),
        n_cooperated=n_coop,
        cooperation_rate=n_coop / len(stories) if stories else 0.0,
        decisions=decisions,
    )


def prune_pass(
    model, tokenizer, layers,
    vector_set: SteeringVectorSet,
    stories: list[dict[str, Any]],
    *,
    alpha_prune: float = 3.0,
    keep_top_k: int = 10,
) -> tuple[list[tuple[int, str]], list[SteeringEvalResult]]:
    """Score every (layer, position) cell at +/- alpha_prune.

    Returns (survivor_keys, all_prune_results). survivor_keys is the top-k
    cells by (coop_rate(+alpha_prune) - coop_rate(-alpha_prune)).
    """
    results: list[SteeringEvalResult] = []
    scores: dict[tuple[int, str], float] = {}
    for key, vec in sorted(vector_set.vectors.items()):
        r_pos = _eval_one_cell(model, tokenizer, layers, vec, +alpha_prune, stories)
        r_neg = _eval_one_cell(model, tokenizer, layers, vec, -alpha_prune, stories)
        results.extend([r_pos, r_neg])
        scores[key] = r_pos.cooperation_rate - r_neg.cooperation_rate

    survivors = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:keep_top_k]
    return [k for k, _ in survivors], results


def sweep_pass(
    model, tokenizer, layers,
    vector_set: SteeringVectorSet,
    survivor_keys: list[tuple[int, str]],
    stories: list[dict[str, Any]],
    *,
    alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
) -> list[SteeringEvalResult]:
    """Full alpha sweep across the surviving cells."""
    results: list[SteeringEvalResult] = []
    for key in survivor_keys:
        vec = vector_set.vectors[key]
        for alpha in alpha_grid:
            results.append(_eval_one_cell(model, tokenizer, layers, vec, alpha, stories))
    return results
