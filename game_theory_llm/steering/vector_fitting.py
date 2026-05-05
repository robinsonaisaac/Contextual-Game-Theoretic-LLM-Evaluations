"""Mean-difference steering vector fitting.

For each (layer, position) cell, we compute:
    direction = mean(coop_acts) - mean(defect_acts)
    raw_norm  = ||direction||
    direction = direction / raw_norm

with class-balancing (subsample the larger class to the size of the smaller)
and a minimum-sample threshold per cell.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import torch

from .models import ActivationBundle, SteeringVector, SteeringVectorSet
from .storage import hash_corpus


def fit_vectors(
    bundles: Iterable[ActivationBundle],
    *,
    model_name: str,
    min_samples: int = 30,
    seed: int = 0,
) -> SteeringVectorSet:
    """Fit one steering vector per (layer, position) cell.

    Cells with fewer than `min_samples` examples in either class are skipped
    (with a UserWarning). If no defect or no coop examples exist anywhere,
    a ValueError is raised.
    """
    bundles = list(bundles)
    if not any(b.cooperated for b in bundles):
        raise ValueError("Cannot fit: no cooperate examples in corpus")
    if not any(not b.cooperated for b in bundles):
        raise ValueError("Cannot fit: no defect examples in corpus")

    # Collect (layer, position) keys present anywhere.
    cells: set[tuple[int, str]] = set()
    for b in bundles:
        for layer_idx, pos_map in b.activations.items():
            for pos in pos_map:
                cells.add((layer_idx, pos))

    rng = np.random.default_rng(seed)
    vectors: dict[tuple[int, str], SteeringVector] = {}

    for layer, position in sorted(cells):
        coop_acts = [
            b.activations[layer][position] for b in bundles
            if b.cooperated and layer in b.activations and position in b.activations[layer]
        ]
        defect_acts = [
            b.activations[layer][position] for b in bundles
            if (not b.cooperated) and layer in b.activations and position in b.activations[layer]
        ]
        if len(coop_acts) < min_samples or len(defect_acts) < min_samples:
            warnings.warn(
                f"Cell (layer={layer}, position={position!r}) below min_samples: "
                f"n_coop={len(coop_acts)}, n_defect={len(defect_acts)}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Class-balance: subsample the larger class.
        n = min(len(coop_acts), len(defect_acts))
        coop_idx = rng.choice(len(coop_acts), n, replace=False)
        defect_idx = rng.choice(len(defect_acts), n, replace=False)
        coop_stack = torch.stack([coop_acts[i] for i in coop_idx]).float()
        defect_stack = torch.stack([defect_acts[i] for i in defect_idx]).float()

        diff = coop_stack.mean(0) - defect_stack.mean(0)
        raw_norm = float(diff.norm().item())
        direction = (diff / raw_norm).to(torch.float32)

        vectors[(layer, position)] = SteeringVector(
            model_name=model_name,
            layer=layer,
            position=position,
            direction=direction,
            raw_norm=raw_norm,
            n_coop=n,
            n_defect=n,
            fit_metadata={"min_samples": min_samples, "seed": seed},
        )

    # Compute corpus hash for caching/provenance.
    import pandas as pd
    corpus_df = pd.DataFrame(
        [{"story_id": b.story_id, "cooperated": b.cooperated} for b in bundles]
    )
    corpus_hash = hash_corpus(corpus_df)

    return SteeringVectorSet(
        model_name=model_name,
        vectors=vectors,
        corpus_hash=corpus_hash,
    )
