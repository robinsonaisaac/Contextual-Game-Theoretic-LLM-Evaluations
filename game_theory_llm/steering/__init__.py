"""Activation-steering tools for white-box experiments on open-weight models.

This package is GPU/PyTorch-heavy. Install via:

    pip install -e ".[steering]"

Most use is via the local driver `scripts/run_steering.py` and the Modal
functions defined in `modal_app.py`.
"""

from .models import (
    ActivationBundle,
    SteeringVector,
    SteeringVectorSet,
    SteeringEvalResult,
    SteeringRun,
)
from .probing import (
    load_index_and_bundles,
    layer_probe,
    logit_attribution,
    subspace_decomposition,
    direction_decision_correlation,
    residual_probe,
    cross_framing_probe,
    game_rsa,
    game_rsa_all_layers,
    run_full_probe_analysis,
)

__all__ = [
    "ActivationBundle",
    "SteeringVector",
    "SteeringVectorSet",
    "SteeringEvalResult",
    "SteeringRun",
    # probing
    "load_index_and_bundles",
    "layer_probe",
    "logit_attribution",
    "subspace_decomposition",
    "direction_decision_correlation",
    "residual_probe",
    "cross_framing_probe",
    "game_rsa",
    "game_rsa_all_layers",
    "run_full_probe_analysis",
]
