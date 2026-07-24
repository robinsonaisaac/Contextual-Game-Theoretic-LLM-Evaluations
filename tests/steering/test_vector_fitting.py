"""Tests for steering vector fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from game_theory_llm.steering.models import ActivationBundle
from game_theory_llm.steering.vector_fitting import fit_vectors


def _bundle(story_id: str, cooperated: bool, layer: int, position: str,
            vec: torch.Tensor) -> ActivationBundle:
    return ActivationBundle(
        story_id=story_id,
        model_name="test-model",
        decision="A" if cooperated else "B",
        cooperated=cooperated,
        prompt_text="prompt",
        trace_text="trace",
        activations={layer: {position: vec.to(torch.bfloat16)}},
        metadata={},
    )


def test_mean_diff_recovers_known_direction():
    """If coop activations cluster around +e1 and defect around -e1, the fitted
    direction must point along +e1 (unit-normalized)."""
    torch.manual_seed(0)
    hidden = 8
    e1 = torch.zeros(hidden); e1[0] = 1.0
    bundles = []
    for i in range(40):
        bundles.append(_bundle(f"c{i}", True, 5, "last_trace",
                               e1 + 0.05 * torch.randn(hidden)))
        bundles.append(_bundle(f"d{i}", False, 5, "last_trace",
                               -e1 + 0.05 * torch.randn(hidden)))
    vs = fit_vectors(bundles, model_name="test-model", min_samples=10)
    v = vs.vectors[(5, "last_trace")]
    assert v.direction.shape == (hidden,)
    assert torch.isclose(v.direction.norm(), torch.tensor(1.0), atol=1e-5)
    # Direction should be very close to +e1.
    assert v.direction[0].item() > 0.99
    assert v.raw_norm > 0
    assert v.n_coop == 40 and v.n_defect == 40


def test_class_balancing_subsamples_to_smaller_class():
    """If there are 100 coop and 20 defect, both n_coop and n_defect should be 20."""
    torch.manual_seed(1)
    hidden = 4
    bundles = []
    for i in range(100):
        bundles.append(_bundle(f"c{i}", True, 0, "last_prompt",
                               torch.randn(hidden)))
    for i in range(20):
        bundles.append(_bundle(f"d{i}", False, 0, "last_prompt",
                               torch.randn(hidden)))
    vs = fit_vectors(bundles, model_name="test-model", min_samples=10)
    v = vs.vectors[(0, "last_prompt")]
    assert v.n_coop == 20
    assert v.n_defect == 20


def test_below_min_samples_skipped_with_warning():
    """If a cell has fewer than min_samples in either class, it is skipped."""
    torch.manual_seed(2)
    hidden = 4
    bundles = [
        _bundle("c0", True, 0, "last_prompt", torch.randn(hidden)),
        _bundle("d0", False, 0, "last_prompt", torch.randn(hidden)),
        _bundle("d1", False, 0, "last_prompt", torch.randn(hidden)),
    ]
    with pytest.warns(UserWarning, match="below min_samples"):
        vs = fit_vectors(bundles, model_name="test-model", min_samples=5)
    assert (0, "last_prompt") not in vs.vectors


def test_all_one_class_raises():
    """If no examples of one class exist anywhere, fit raises a clear error."""
    bundles = [
        _bundle(f"c{i}", True, 0, "last_prompt", torch.randn(4)) for i in range(10)
    ]
    with pytest.raises(ValueError, match="no defect"):
        fit_vectors(bundles, model_name="test-model", min_samples=5)
