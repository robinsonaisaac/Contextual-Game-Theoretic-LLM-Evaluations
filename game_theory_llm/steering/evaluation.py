"""Two-pass eval: pruning at +/- alpha_max, then full sweep on top-k cells."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
    progress_path: Path | None = None,
    progress_tag: str = "",
) -> SteeringEvalResult:
    """Evaluate one (layer, position, alpha) cell across all eval stories.

    If `progress_path` is set, append one JSONL line per (cell, story) so a
    crashed/disconnected run still preserves partial results.
    """
    decisions = []
    n_coop = 0
    with steering_hook(layers, vec, alpha):
        for s in stories:
            t0 = time.time()
            text = generate_with_hook(
                model, tokenizer, s["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=s.get("seed"),
                apply_chat_template=s.get("apply_chat_template", True),
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
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a") as f:
                    f.write(json.dumps({
                        "tag": progress_tag,
                        "layer": vec.layer,
                        "position": vec.position,
                        "alpha": alpha,
                        "story_id": s["story_id"],
                        "decision": d,
                        "cooperated": cooperated,
                        "elapsed_s": time.time() - t0,
                        "trace": text,
                    }) + "\n")
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
    progress_path: Path | None = None,
) -> tuple[list[tuple[int, str]], list[SteeringEvalResult]]:
    """Score every (layer, position) cell at +/- alpha_prune.

    Returns (survivor_keys, all_prune_results). survivor_keys is the top-k
    cells by (coop_rate(+alpha_prune) - coop_rate(-alpha_prune)).
    """
    results: list[SteeringEvalResult] = []
    scores: dict[tuple[int, str], float] = {}
    n_cells = len(vector_set.vectors)
    for i, (key, vec) in enumerate(sorted(vector_set.vectors.items()), 1):
        print(f"[prune] cell {i}/{n_cells}: layer={key[0]} position={key[1]}", flush=True)
        r_pos = _eval_one_cell(model, tokenizer, layers, vec, +alpha_prune, stories,
                               progress_path=progress_path, progress_tag="prune")
        r_neg = _eval_one_cell(model, tokenizer, layers, vec, -alpha_prune, stories,
                               progress_path=progress_path, progress_tag="prune")
        results.extend([r_pos, r_neg])
        scores[key] = r_pos.cooperation_rate - r_neg.cooperation_rate
        print(f"[prune]   +α={r_pos.cooperation_rate:.2f}  -α={r_neg.cooperation_rate:.2f}  "
              f"Δ={scores[key]:+.2f}", flush=True)

    survivors = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:keep_top_k]
    return [k for k, _ in survivors], results


def sweep_pass(
    model, tokenizer, layers,
    vector_set: SteeringVectorSet,
    survivor_keys: list[tuple[int, str]],
    stories: list[dict[str, Any]],
    *,
    alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    progress_path: Path | None = None,
) -> list[SteeringEvalResult]:
    """Full alpha sweep across the surviving cells."""
    results: list[SteeringEvalResult] = []
    for j, key in enumerate(survivor_keys, 1):
        vec = vector_set.vectors[key]
        for alpha in alpha_grid:
            print(f"[sweep] cell {j}/{len(survivor_keys)} layer={key[0]} "
                  f"position={key[1]} alpha={alpha:+.1f}", flush=True)
            r = _eval_one_cell(model, tokenizer, layers, vec, alpha, stories,
                               progress_path=progress_path, progress_tag="sweep")
            print(f"[sweep]   coop_rate={r.cooperation_rate:.2f}", flush=True)
            results.append(r)
    return results
