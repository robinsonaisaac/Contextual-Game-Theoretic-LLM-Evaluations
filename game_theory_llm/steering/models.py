"""Data classes for the activation-steering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


POSITION_KEYS = ("last_prompt", "last_trace", "mean_trace")


@dataclass
class ActivationBundle:
    """Activations captured for a single story under a single model."""
    story_id: str
    model_name: str            # e.g. "gemma-4-e4b-it"
    decision: str              # parsed from the trace ("A" or "B")
    cooperated: bool           # decision == cooperative choice for this PD
    prompt_text: str
    trace_text: str
    # activations[layer_idx][position_key] -> 1D bf16 tensor of size hidden_dim
    activations: dict[int, dict[str, torch.Tensor]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SteeringVector:
    """A single steering direction for one (layer, position)."""
    model_name: str
    layer: int
    position: str              # one of POSITION_KEYS
    direction: torch.Tensor    # 1D fp32, hidden_dim, unit-normalized
    raw_norm: float            # ||mean_coop - mean_defect|| pre-normalization
    n_coop: int
    n_defect: int
    fit_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SteeringVectorSet:
    """All (layer, position) vectors fit from one corpus."""
    model_name: str
    vectors: dict[tuple[int, str], SteeringVector]
    corpus_hash: str
    fit_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SteeringEvalResult:
    """Cooperation rate for one (layer, position, alpha) cell on held-out stories."""
    layer: int
    position: str
    alpha: float
    n_stories: int
    n_cooperated: int
    cooperation_rate: float
    decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SteeringRun:
    """Manifest tying corpus -> vectors -> eval results for one experiment."""
    run_id: str
    model_name: str
    train_story_ids: list[str]
    eval_story_ids: list[str]
    vector_set_path: str
    results_path: str
    config: dict[str, Any] = field(default_factory=dict)
