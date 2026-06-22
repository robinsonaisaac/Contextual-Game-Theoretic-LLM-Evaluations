"""Name shortlisted features by their max-activating examples.

Residuals for all texts come back in ONE remote call per layer; SAE-encode is local.
The caller is responsible for batching: pass all texts at once and reuse the result
for every feature at the same layer.
"""
from __future__ import annotations

import numpy as np
import torch

from . import remote


def max_activating_examples(
    layer: int,
    feature: int,
    texts: list[str],
    sae,
    topn: int = 8,
) -> list[dict]:
    """Return the *topn* texts with the highest activation of *feature* at *layer*.

    Residuals are fetched from the Modal worker in a single batch call.  The SAE
    encoding is done locally on the returned numpy array.

    Args:
        layer:   transformer layer index (must match a downloaded SAE).
        feature: SAE feature index (0 … D_SAE-1).
        texts:   pool of input texts.
        sae:     a loaded QwenScopeSAE instance for *layer*.
        topn:    how many top-activating examples to return.

    Returns:
        List of ``{"score": float, "snippet": str}`` dicts ordered descending by score.
    """
    R = remote.extract_residuals(texts, [layer])           # [N, 1, D_MODEL] numpy
    R_t = torch.tensor(R[:, 0, :], dtype=torch.float32)   # [N, D_MODEL]
    acts = sae.encode(R_t)                                 # [N, D_SAE]
    col = acts[:, feature]                                 # [N]
    k = min(topn, len(texts))
    order = torch.argsort(col, descending=True)[:k].tolist()
    return [{"score": float(col[i].item()), "snippet": texts[i][:300]} for i in order]


def batch_max_activating(
    layer: int,
    features: list[int],
    texts: list[str],
    sae,
    topn: int = 8,
) -> dict[int, list[dict]]:
    """Fetch residuals ONCE for *layer* and return max-activating examples for
    every feature in *features*.  Much cheaper than calling max_activating_examples
    per feature.

    Returns:
        ``{feature_id: [{"score": float, "snippet": str}, ...]}``
    """
    R = remote.extract_residuals(texts, [layer])           # [N, 1, D_MODEL] numpy
    R_t = torch.tensor(R[:, 0, :], dtype=torch.float32)   # [N, D_MODEL]
    acts = sae.encode(R_t)                                 # [N, D_SAE]

    result: dict[int, list[dict]] = {}
    for feature in features:
        col = acts[:, feature]
        k = min(topn, len(texts))
        order = torch.argsort(col, descending=True)[:k].tolist()
        result[feature] = [
            {"score": float(col[i].item()), "snippet": texts[i][:300]}
            for i in order
        ]
    return result
