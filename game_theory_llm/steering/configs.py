"""Per-model steering configuration.

The pipeline is parameterized by these configs so a single orchestrator
script (``scripts/run_full_pipeline.py``) can run the same workflow on
each Gemma 4 variant. Architecture facts (n_layers, hidden_dim) come from
``modal_app.py::warmup`` — see comments in that file.

`candidate_layers` lists the (layer, position) cells we want to test in
the eval shard sweep. They were chosen from our pilot results:

- E4B: L16 mean_trace gave the cleanest 60pp dose-response curve in our
  cross-layer 2D sweep. L25 mean_trace was the highest raw_norm but
  weaker in causal effect — kept here as a sanity check.
- E2B: L17 mean_trace gave a 46pp range; L13 was weaker (20pp) but kept
  for breadth.
- 26B-A4B: L11 mean_trace was massive (94pp range, near saturation at
  α=±3). L16 was a moderate 47pp range, kept as a comparison.

`gpu_tier` selects the worker class:
- "small" -> SteeringWorker (A100-40GB), suitable for ≤7B models.
- "large" -> SteeringWorkerLarge (A100-80GB), required for ≥10B models.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    short_name: str
    hf_id: str
    gpu_tier: str             # "small" | "large"
    n_layers: int
    candidate_layers: Tuple[int, ...] = ()
    multi_cells: Tuple[Tuple[Tuple[int, str], ...], ...] = ()


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "E2B": ModelConfig(
        short_name="E2B",
        hf_id="google/gemma-4-E2B-it",
        gpu_tier="small",
        n_layers=35,
        candidate_layers=(13, 17),
    ),
    "E4B": ModelConfig(
        short_name="E4B",
        hf_id="google/gemma-4-E4B-it",
        gpu_tier="small",
        n_layers=42,
        candidate_layers=(16, 25),
        multi_cells=(((25, "mean_trace"), (27, "mean_trace")),),
    ),
    "26B-A4B": ModelConfig(
        short_name="26B-A4B",
        hf_id="google/gemma-4-26B-A4B-it",
        gpu_tier="large",
        n_layers=30,
        candidate_layers=(11, 16),
    ),
}


def get_config(name: str) -> ModelConfig:
    """Lookup by short_name with friendly error if missing."""
    if name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[name]


def run_id_for(model_short_name: str, tag: str = "v1") -> str:
    """Stable run_id used everywhere on the volume for a given model."""
    return f"pd_{model_short_name.replace('-', '_')}_{tag}"
