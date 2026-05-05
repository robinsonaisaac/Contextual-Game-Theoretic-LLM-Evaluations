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

__all__ = [
    "ActivationBundle",
    "SteeringVector",
    "SteeringVectorSet",
    "SteeringEvalResult",
    "SteeringRun",
]
