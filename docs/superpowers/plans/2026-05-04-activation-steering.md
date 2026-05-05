# Activation Steering Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `game_theory_llm/steering/` package that extracts residual-stream activations from Gemma 4 E4B-it on Modal, fits a mean-difference steering vector for cooperation, and validates it via an α-sweep on held-out PD stories.

**Architecture:** New self-contained package alongside the existing OpenRouter-based code. White-box code (PyTorch hooks, HuggingFace `transformers`) lives only in this package, gated behind a `[steering]` extras install. Modal runs extraction and evaluation; the local driver does fitting and orchestration. Three token positions × 42 layers × seven α values are explored via prune-then-sweep.

**Tech Stack:** PyTorch 2.5, HuggingFace `transformers` ≥4.46, Modal, pyarrow/Parquet, the existing `game_theory_llm` story generation pipeline.

**Spec:** `docs/superpowers/specs/2026-05-04-activation-steering-design.md`

**Testing philosophy:** Per the spec, only `vector_fitting.py` gets unit tests (it is pure math, easy to break silently). Other modules are smoke-tested via `warmup()` and the end-to-end tiny experiment. We do NOT TDD modules whose behavior is dominated by GPU/model integration; we write them, run smoke checks, and iterate.

---

## Setup (one-time, not a task)

Before starting Task 1, the implementer should:

1. `pip install modal`
2. `modal token new` (one-time browser auth — opens a browser tab; output ends with `Token verified successfully!`)
3. `python3 -c "import modal; print(modal.__version__)"` to confirm install (expect ≥0.66)
4. Confirm `OPENROUTER_API_KEY` is in `.env` (existing project convention, used for story generation but not for steering itself)

---

## Task 1: Add steering package skeleton and dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `game_theory_llm/steering/__init__.py`
- Create: `tests/steering/__init__.py`

- [ ] **Step 1: Add `[steering]` extras to `pyproject.toml`**

Edit the `[project.optional-dependencies]` section to add a `steering` extra. The full updated section should look like:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "ruff",
]
ml = [
    "xgboost",
    "scikit-learn",
    "sentence-transformers",
]
steering = [
    "torch>=2.5,<3",
    "transformers>=4.46",
    "accelerate>=1.0",
    "huggingface_hub",
    "pyarrow>=14",
    "modal>=0.66",
]
```

- [ ] **Step 2: Create the package init**

Create `game_theory_llm/steering/__init__.py` with this content:

```python
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
```

- [ ] **Step 3: Create the test package init**

Create `tests/steering/__init__.py` as an empty file:

```python
```

- [ ] **Step 4: Install the new extra**

Run: `pip install -e ".[steering]"`
Expected: pip resolves and installs `torch`, `transformers`, `modal`, etc.

- [ ] **Step 5: Verify imports work**

Run: `python3 -c "import torch, transformers, modal; print(torch.__version__, transformers.__version__, modal.__version__)"`
Expected: prints three version strings, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml game_theory_llm/steering/__init__.py tests/steering/__init__.py
git commit -m "feat(steering): add steering package skeleton and [steering] extras"
```

---

## Task 2: Define the data model

**Files:**
- Create: `game_theory_llm/steering/models.py`

- [ ] **Step 1: Write `models.py`**

Create `game_theory_llm/steering/models.py` with these dataclasses:

```python
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
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from game_theory_llm.steering import ActivationBundle, SteeringVector, SteeringVectorSet, SteeringEvalResult, SteeringRun; print('ok')"`
Expected: prints `ok`, no errors.

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/models.py
git commit -m "feat(steering): add data model dataclasses"
```

---

## Task 3: Storage helpers (per-story .pt + Parquet index)

**Files:**
- Create: `game_theory_llm/steering/storage.py`

- [ ] **Step 1: Write `storage.py`**

Create `game_theory_llm/steering/storage.py`:

```python
"""Serialization for activations, vector sets, and eval results.

Activations are stored one .pt file per story (resumable, easy to inspect).
A Parquet index records which file holds which story so fitting code can
iterate without loading everything into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .models import (
    ActivationBundle,
    SteeringEvalResult,
    SteeringVectorSet,
)


# ---------- ActivationBundle ----------

def save_activation_bundle(bundle: ActivationBundle, dest_dir: Path) -> Path:
    """Write one ActivationBundle to {dest_dir}/{story_id}.pt and return the path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{bundle.story_id}.pt"
    torch.save(
        {
            "story_id": bundle.story_id,
            "model_name": bundle.model_name,
            "decision": bundle.decision,
            "cooperated": bundle.cooperated,
            "prompt_text": bundle.prompt_text,
            "trace_text": bundle.trace_text,
            "activations": bundle.activations,
            "metadata": bundle.metadata,
        },
        path,
    )
    return path


def load_activation_bundle(path: Path) -> ActivationBundle:
    """Load one ActivationBundle from a .pt file."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return ActivationBundle(**blob)


# ---------- Index Parquet ----------

def write_index(bundles: Iterable[ActivationBundle], paths: Iterable[Path],
                index_path: Path, split: str) -> None:
    """Write/append a Parquet index with one row per (story, split)."""
    rows = [
        {
            "story_id": b.story_id,
            "decision": b.decision,
            "cooperated": b.cooperated,
            "split": split,
            "path": str(p),
            "model_name": b.model_name,
        }
        for b, p in zip(bundles, paths)
    ]
    new_df = pd.DataFrame(rows)
    if index_path.exists():
        existing = pd.read_parquet(index_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["story_id", "split"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(index_path, index=False)


def read_index(index_path: Path, split: str | None = None) -> pd.DataFrame:
    """Read the index Parquet, optionally filtered by split."""
    df = pd.read_parquet(index_path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


# ---------- Corpus hashing ----------

def hash_corpus(rows: pd.DataFrame) -> str:
    """Stable hash over (story_id, cooperated) tuples; used as corpus_hash."""
    pairs = sorted(
        (str(r.story_id), bool(r.cooperated)) for r in rows.itertuples(index=False)
    )
    h = hashlib.sha256()
    for sid, coop in pairs:
        h.update(sid.encode())
        h.update(b"\x01" if coop else b"\x00")
    return h.hexdigest()[:16]


# ---------- SteeringVectorSet ----------

def save_vector_set(vs: SteeringVectorSet, path: Path) -> None:
    """Save a SteeringVectorSet to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": vs.model_name,
            "vectors": {
                f"{k[0]}__{k[1]}": {
                    "model_name": v.model_name,
                    "layer": v.layer,
                    "position": v.position,
                    "direction": v.direction,
                    "raw_norm": v.raw_norm,
                    "n_coop": v.n_coop,
                    "n_defect": v.n_defect,
                    "fit_metadata": v.fit_metadata,
                }
                for k, v in vs.vectors.items()
            },
            "corpus_hash": vs.corpus_hash,
            "fit_timestamp": vs.fit_timestamp,
        },
        path,
    )


def load_vector_set(path: Path) -> SteeringVectorSet:
    """Load a SteeringVectorSet."""
    from .models import SteeringVector  # avoid circular import at module level
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vectors = {}
    for k, v in blob["vectors"].items():
        layer_str, position = k.split("__", 1)
        vectors[(int(layer_str), position)] = SteeringVector(**v)
    return SteeringVectorSet(
        model_name=blob["model_name"],
        vectors=vectors,
        corpus_hash=blob["corpus_hash"],
        fit_timestamp=blob["fit_timestamp"],
    )


# ---------- Eval results ----------

def save_eval_results(results: list[SteeringEvalResult], path: Path) -> None:
    """Flatten eval results to a Parquet with one row per (cell, story)."""
    rows = []
    for r in results:
        for d in r.decisions:
            rows.append({
                "layer": r.layer,
                "position": r.position,
                "alpha": r.alpha,
                "story_id": d["story_id"],
                "decision": d["decision"],
                "cooperated": d["cooperated"],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from game_theory_llm.steering.storage import save_activation_bundle, load_activation_bundle, write_index, read_index, hash_corpus, save_vector_set, load_vector_set, save_eval_results; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/storage.py
git commit -m "feat(steering): add storage helpers (.pt bundles + Parquet index)"
```

---

## Task 4: Modal app skeleton and warmup discovery function

**Files:**
- Create: `game_theory_llm/steering/modal_app.py`

- [ ] **Step 1: Write `modal_app.py` with the `warmup` function**

Create `game_theory_llm/steering/modal_app.py`:

```python
"""Modal app for activation extraction and steering evaluation on Gemma 4 E4B-it.

Run a one-shot warmup discovery from your laptop:

    modal run game_theory_llm/steering/modal_app.py::warmup

This loads the model on an A100, attaches a no-op forward hook, and returns
the actual layer count, hidden dim, module path, and dtype. Use the returned
numbers to populate downstream extraction code.
"""

from __future__ import annotations

import modal

MODEL_NAME = "google/gemma-4-E4B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface_hub",
        "pandas>=2.0",
        "pyarrow>=14",
        "numpy>=1.24",
    )
    .add_local_python_source("game_theory_llm")
)

volume = modal.Volume.from_name("gtllm-steering", create_if_missing=True)

app = modal.App("gtllm-steering", image=image)


@app.function(gpu="A100", timeout=600)
def warmup() -> dict:
    """Discover model architecture by loading once and running a tiny forward.

    Returns a dict with: n_layers, hidden_dim, module_path, dtype, vocab_size.
    Also confirms forward_hook attachment works (returns hook_ok=True).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Discover module path. Try the two most likely candidates.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        module_path = "model.model.layers"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
        module_path = "model.language_model.layers"
    else:
        # Walk the module tree to find a list of identical decoder blocks.
        layers = None
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                layers = mod
                module_path = name
                break
        if layers is None:
            raise RuntimeError("Could not locate decoder layer list")

    n_layers = len(layers)
    hidden_dim = model.config.hidden_size

    # Confirm hook attachment works on a real forward.
    captured = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(h.shape)
        captured["dtype"] = str(h.dtype)
        return output  # no-op
    handle = layers[n_layers // 2].register_forward_hook(hook)
    try:
        ids = tokenizer("Hello, world.", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    return {
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "vocab_size": model.config.vocab_size,
        "module_path": module_path,
        "dtype": "bfloat16",
        "hook_ok": "shape" in captured,
        "hook_capture_shape": captured.get("shape"),
        "hook_capture_dtype": captured.get("dtype"),
    }
```

- [ ] **Step 2: Run `warmup` on Modal**

Run: `modal run game_theory_llm/steering/modal_app.py::warmup`

Expected output (numbers may vary slightly): a dict containing roughly
```
{'model_name': 'google/gemma-4-E4B-it',
 'n_layers': 42,
 'hidden_dim': <int>,
 'vocab_size': 262144,
 'module_path': 'model.model.layers',
 'dtype': 'bfloat16',
 'hook_ok': True,
 'hook_capture_shape': (1, <seq_len>, <hidden_dim>),
 'hook_capture_dtype': 'torch.bfloat16'}
```

- [ ] **Step 3: Record the discovered numbers**

Take the printed `n_layers`, `hidden_dim`, and `module_path` from Step 2 and add them as a comment block immediately below the existing module docstring in `modal_app.py`. Use today's date and the actual numbers from the run:

```python
# Discovered by warmup() on 2026-MM-DD:
#   n_layers     = 42        # actual value from warmup()
#   hidden_dim   = <actual>
#   module_path  = "model.model.layers"  # actual value from warmup()
#   dtype        = bfloat16
```

If `module_path` came back different from `model.model.layers`, every subsequent task that references that string MUST use the discovered value. Track this carefully.

- [ ] **Step 4: Commit**

```bash
git add game_theory_llm/steering/modal_app.py
git commit -m "feat(steering): add Modal app and warmup discovery function"
```

---

## Task 5: Extraction module

**Files:**
- Create: `game_theory_llm/steering/extraction.py`

- [ ] **Step 1: Write `extraction.py`**

Create `game_theory_llm/steering/extraction.py`:

```python
"""Activation extraction: generate a trace, then re-run with hooks to capture
residual-stream activations at three token positions for every layer.

Designed to be called from inside a Modal worker that has already loaded the
model. The functions here take `model`, `tokenizer`, and the list of layer
modules so they are agnostic to the exact module path.
"""

from __future__ import annotations

from typing import Any

import torch

from game_theory_llm.decision_parser import extract_decision

from .models import POSITION_KEYS


def generate_trace(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int | None = None,
) -> tuple[str, torch.Tensor]:
    """Sample a reasoning trace given a prompt.

    Returns (trace_text, full_token_ids), where full_token_ids is the
    concatenation of prompt and generated tokens (1D tensor on CPU).
    """
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_ids = out[0].detach().cpu()
    trace_ids = full_ids[prompt_len:]
    trace_text = tokenizer.decode(trace_ids, skip_special_tokens=True)
    return trace_text, full_ids


def extract_activations(
    model,
    tokenizer,
    layers: torch.nn.ModuleList,
    prompt: str,
    trace_text: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Re-run prompt+trace through the model with hooks attached to every
    decoder block, capturing the residual stream at three positions.

    Returns activations[layer_idx][position_key] -> 1D bf16 tensor on CPU.
    """
    device = model.device
    full_text = prompt + trace_text
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    seq_len = full_ids.shape[1]
    if seq_len <= prompt_len:
        # Trace was empty or tokenizer collapsed it; capture only last_prompt.
        prompt_len = seq_len - 1
    last_prompt_idx = prompt_len - 1
    last_trace_idx = seq_len - 1
    trace_slice = slice(prompt_len, seq_len)

    # Storage for captured activations.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (batch=1, seq, hidden); detach + move to CPU + bf16.
            captured[layer_idx] = h[0].detach().to("cpu", torch.bfloat16)
            return output
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            model(full_ids)
    finally:
        for h in handles:
            h.remove()

    activations: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, h in captured.items():
        activations[layer_idx] = {
            "last_prompt": h[last_prompt_idx].clone(),
            "last_trace": h[last_trace_idx].clone(),
            "mean_trace": h[trace_slice].float().mean(0).to(torch.bfloat16),
        }
    return activations


def parse_decision(trace_text: str) -> str | None:
    """Wrap the existing extract_decision; returns 'A', 'B', or None."""
    return extract_decision(trace_text)


def is_cooperative(decision: str | None, coop_choice: str) -> bool:
    """Whether a decision matches the cooperative choice for this PD.

    coop_choice is the letter ('A' or 'B') the payoff matrix designates as
    the cooperative play. Returns False on None.
    """
    if decision is None:
        return False
    return decision == coop_choice
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from game_theory_llm.steering.extraction import generate_trace, extract_activations, parse_decision, is_cooperative; print('ok')"`
Expected: `ok`. (Imports succeed even though the functions need a real model — we are only checking module health.)

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/extraction.py
git commit -m "feat(steering): add activation extraction (generate + hooked re-run)"
```

---

## Task 6: Vector fitting (TDD)

**Files:**
- Create: `tests/steering/test_vector_fitting.py`
- Create: `game_theory_llm/steering/vector_fitting.py`

This is the only module with rigorous TDD per the spec — pure math, easy to break silently.

- [ ] **Step 1: Write the failing tests**

Create `tests/steering/test_vector_fitting.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/steering/test_vector_fitting.py -v`
Expected: ALL tests fail with ImportError or ModuleNotFoundError on `game_theory_llm.steering.vector_fitting`.

- [ ] **Step 3: Implement `vector_fitting.py`**

Create `game_theory_llm/steering/vector_fitting.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/steering/test_vector_fitting.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/steering/test_vector_fitting.py game_theory_llm/steering/vector_fitting.py
git commit -m "feat(steering): add mean-diff vector fitting with class balancing (tested)"
```

---

## Task 7: Steering application (hook factory)

**Files:**
- Create: `game_theory_llm/steering/application.py`

- [ ] **Step 1: Write `application.py`**

Create `game_theory_llm/steering/application.py`:

```python
"""Apply a steering vector at inference time via a forward hook on one layer.

Usage (inside a Modal worker that has model + tokenizer + layers):

    with steering_hook(layers, vec, alpha=1.0) as ctx:
        text = generate_with_hook(model, tokenizer, prompt)
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .models import SteeringVector


def make_steering_hook(direction: torch.Tensor, alpha: float, raw_norm: float):
    """Build a forward hook that adds (alpha * raw_norm) * direction to the
    layer output's residual stream at every token position."""
    scaled = (alpha * raw_norm) * direction
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + scaled.to(h.device, h.dtype)
            return (h, *output[1:])
        return output + scaled.to(output.device, output.dtype)
    return hook


@contextmanager
def steering_hook(layers: torch.nn.ModuleList, vec: SteeringVector, alpha: float):
    """Context manager: attach the steering hook on entry, remove on exit."""
    handle = layers[vec.layer].register_forward_hook(
        make_steering_hook(vec.direction, alpha, vec.raw_norm)
    )
    try:
        yield handle
    finally:
        handle.remove()


def generate_with_hook(model, tokenizer, prompt: str, *,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       seed: int | None = None) -> str:
    """Generate a trace from prompt; the caller is responsible for any active hooks."""
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from game_theory_llm.steering.application import make_steering_hook, steering_hook, generate_with_hook; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/application.py
git commit -m "feat(steering): add hook factory and generate-with-steering helper"
```

---

## Task 8: Evaluation (prune-then-sweep)

**Files:**
- Create: `game_theory_llm/steering/evaluation.py`

- [ ] **Step 1: Write `evaluation.py`**

Create `game_theory_llm/steering/evaluation.py`:

```python
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
```

- [ ] **Step 2: Verify imports**

Run: `python3 -c "from game_theory_llm.steering.evaluation import prune_pass, sweep_pass; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/evaluation.py
git commit -m "feat(steering): add prune-then-sweep evaluation"
```

---

## Task 9: Modal worker class wiring extraction + evaluation

**Files:**
- Modify: `game_theory_llm/steering/modal_app.py`

- [ ] **Step 1: Add the SteeringWorker class to `modal_app.py`**

Append the following to `game_theory_llm/steering/modal_app.py` (after the existing `warmup` function):

```python
@app.cls(
    gpu="A100",
    volumes={"/data": volume},
    timeout=3600,
    scaledown_window=300,
)
class SteeringWorker:
    """Long-lived Modal container that loads Gemma 4 E4B-it once."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        # Discover layers (mirror warmup logic).
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            self.layers = self.model.language_model.layers
        else:
            for _, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                    self.layers = mod
                    break
            else:
                raise RuntimeError("Could not locate decoder layer list")

    @modal.method()
    def extract(self, stories: list[dict], run_id: str, split: str) -> dict:
        """Extract activations for a list of stories. Each story dict needs:
            story_id, prompt, coop_choice, (optional) seed, temperature.
        Bundles are written to /data/runs/{run_id}/activations/{split}/.
        """
        from pathlib import Path
        import pandas as pd
        from game_theory_llm.steering.extraction import (
            generate_trace, extract_activations, parse_decision, is_cooperative,
        )
        from game_theory_llm.steering.models import ActivationBundle
        from game_theory_llm.steering.storage import save_activation_bundle, write_index

        run_dir = Path(f"/data/runs/{run_id}")
        bundle_dir = run_dir / "activations" / split
        index_path = run_dir / "index.parquet"

        bundles, paths = [], []
        for s in stories:
            trace_text, _ = generate_trace(
                self.model, self.tokenizer, s["prompt"],
                max_new_tokens=s.get("max_new_tokens", 512),
                temperature=s.get("temperature", 0.7),
                seed=s.get("seed"),
            )
            decision = parse_decision(trace_text)
            cooperated = is_cooperative(decision, s["coop_choice"])
            acts = extract_activations(
                self.model, self.tokenizer, self.layers,
                s["prompt"], trace_text,
            )
            bundle = ActivationBundle(
                story_id=s["story_id"],
                model_name=MODEL_NAME,
                decision=decision or "",
                cooperated=cooperated,
                prompt_text=s["prompt"],
                trace_text=trace_text,
                activations=acts,
                metadata={
                    "temperature": s.get("temperature", 0.7),
                    "seed": s.get("seed"),
                    "coop_choice": s["coop_choice"],
                },
            )
            path = save_activation_bundle(bundle, bundle_dir)
            bundles.append(bundle)
            paths.append(path)

        write_index(bundles, paths, index_path, split=split)
        # Return summary only (don't ship bundles back over the wire).
        return {
            "n_stories": len(bundles),
            "decision_counts": pd.Series([b.decision for b in bundles]).value_counts().to_dict(),
            "coop_rate": sum(b.cooperated for b in bundles) / max(1, len(bundles)),
            "index_path": str(index_path),
        }

    @modal.method()
    def evaluate(self, run_id: str, stories: list[dict],
                 alpha_prune: float = 3.0,
                 keep_top_k: int = 10,
                 alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)) -> dict:
        """Run prune-then-sweep evaluation against the SteeringVectorSet stored
        at /data/runs/{run_id}/vectors.pt. Eval stories carry the same fields
        as extract() inputs."""
        from pathlib import Path
        from game_theory_llm.steering.evaluation import prune_pass, sweep_pass
        from game_theory_llm.steering.storage import load_vector_set, save_eval_results

        run_dir = Path(f"/data/runs/{run_id}")
        vs = load_vector_set(run_dir / "vectors.pt")

        survivors, prune_results = prune_pass(
            self.model, self.tokenizer, self.layers, vs, stories,
            alpha_prune=alpha_prune, keep_top_k=keep_top_k,
        )
        sweep_results = sweep_pass(
            self.model, self.tokenizer, self.layers, vs, survivors, stories,
            alpha_grid=tuple(alpha_grid),
        )

        prune_path = run_dir / "results_prune.parquet"
        sweep_path = run_dir / "results_sweep.parquet"
        save_eval_results(prune_results, prune_path)
        save_eval_results(sweep_results, sweep_path)

        return {
            "n_survivors": len(survivors),
            "survivors": survivors,
            "prune_path": str(prune_path),
            "sweep_path": str(sweep_path),
        }
```

- [ ] **Step 2: Verify the Modal app still parses**

Run: `modal run game_theory_llm/steering/modal_app.py::warmup --help`
Expected: prints Modal CLI help text for the warmup function. (This implicitly imports the whole file, catching syntax/decorator errors.)

- [ ] **Step 3: Commit**

```bash
git add game_theory_llm/steering/modal_app.py
git commit -m "feat(steering): add SteeringWorker class with extract and evaluate methods"
```

---

## Task 10: Local driver script

**Files:**
- Create: `scripts/run_steering.py`

- [ ] **Step 1: Write the driver**

Create `scripts/run_steering.py`:

```python
"""Local driver for the steering pipeline.

Usage:
    python3 scripts/run_steering.py extract --run-id run1 --stories train.jsonl --split train
    python3 scripts/run_steering.py extract --run-id run1 --stories eval.jsonl  --split eval
    python3 scripts/run_steering.py fit     --run-id run1
    python3 scripts/run_steering.py eval    --run-id run1 --stories eval.jsonl

The `train.jsonl` / `eval.jsonl` files are line-delimited JSON with fields:
    story_id, prompt, coop_choice (one of "A"/"B"), optional seed, temperature.

`extract` and `eval` ship the stories to Modal. `fit` runs locally on the
activation bundles downloaded from the Modal Volume (or a local copy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_extract(args):
    import modal
    from game_theory_llm.steering.modal_app import app, SteeringWorker

    stories = _read_jsonl(Path(args.stories))
    with app.run():
        worker = SteeringWorker()
        summary = worker.extract.remote(stories, args.run_id, args.split)
    print(json.dumps(summary, indent=2))


def cmd_fit(args):
    """Fit vectors locally from per-story bundles in a Modal Volume snapshot.

    Requires that you've previously synced the Modal Volume to local disk:
        modal volume get gtllm-steering runs/{run_id}/ ./local_data/runs/{run_id}/
    """
    from game_theory_llm.steering.storage import (
        load_activation_bundle, read_index, save_vector_set,
    )
    from game_theory_llm.steering.vector_fitting import fit_vectors

    local_run_dir = Path(args.local_dir) / "runs" / args.run_id
    index_path = local_run_dir / "index.parquet"
    df = read_index(index_path, split="train")

    bundles = []
    for row in df.itertuples(index=False):
        bundle = load_activation_bundle(Path(row.path))
        bundles.append(bundle)

    vs = fit_vectors(bundles, model_name=args.model_name)
    out_path = local_run_dir / "vectors.pt"
    save_vector_set(vs, out_path)
    print(json.dumps({
        "n_vectors": len(vs.vectors),
        "corpus_hash": vs.corpus_hash,
        "vectors_path": str(out_path),
    }, indent=2))


def cmd_push_vectors(args):
    """Push the locally-fit vectors.pt into the Modal Volume."""
    import subprocess
    src = Path(args.local_dir) / "runs" / args.run_id / "vectors.pt"
    dst = f"runs/{args.run_id}/vectors.pt"
    if not src.exists():
        sys.exit(f"vectors.pt not found at {src}")
    subprocess.check_call(["modal", "volume", "put", "--force",
                            "gtllm-steering", str(src), dst])
    print(f"Uploaded {src} -> volume:{dst}")


def cmd_eval(args):
    import modal
    from game_theory_llm.steering.modal_app import app, SteeringWorker

    stories = _read_jsonl(Path(args.stories))
    with app.run():
        worker = SteeringWorker()
        summary = worker.evaluate.remote(
            stories,
            run_id=args.run_id,
            alpha_prune=args.alpha_prune,
            keep_top_k=args.keep_top_k,
            alpha_grid=tuple(args.alpha_grid),
        )
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--run-id", required=True)
    ext.add_argument("--stories", required=True)
    ext.add_argument("--split", required=True, choices=["train", "eval"])
    ext.set_defaults(func=cmd_extract)

    fit = sub.add_parser("fit")
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--local-dir", default="./local_data")
    fit.add_argument("--model-name", default="gemma-4-e4b-it")
    fit.set_defaults(func=cmd_fit)

    push = sub.add_parser("push-vectors")
    push.add_argument("--run-id", required=True)
    push.add_argument("--local-dir", default="./local_data")
    push.set_defaults(func=cmd_push_vectors)

    ev = sub.add_parser("eval")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--stories", required=True)
    ev.add_argument("--alpha-prune", type=float, default=3.0)
    ev.add_argument("--keep-top-k", type=int, default=10)
    ev.add_argument("--alpha-grid", type=float, nargs="+",
                    default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ev.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the driver parses and shows help**

Run: `python3 scripts/run_steering.py --help`
Expected: prints subcommand list (`extract`, `fit`, `push-vectors`, `eval`).

Run: `python3 scripts/run_steering.py extract --help`
Expected: prints `--run-id`, `--stories`, `--split` options.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_steering.py
git commit -m "feat(steering): add local driver script for extract / fit / eval"
```

---

## Task 11: End-to-end tiny smoke run

> **Deferred.** This task is blocked on the story-generation pipeline, which is currently being edited. Run this only after the pipeline is stable and at least three real stories are available in the format the driver expects. A small story-format bridge helper (between the existing `Story` objects and the steering JSONL) will be added as its own task at that time.

**Goal:** Verify the full pipeline runs on three stories before doing a real experiment. Costs ~$0.50.

**Files:**
- Create: `scripts/_smoke_stories.jsonl` (gitignored — see Step 1)
- Create: `.gitignore` entry for `local_data/` and `scripts/_smoke_*.jsonl`

- [ ] **Step 1: Add gitignore entries**

Append to `.gitignore` (or create if missing):

```
local_data/
scripts/_smoke_*.jsonl
```

- [ ] **Step 2: Hand-craft three smoke stories**

Generate three stories using the existing `StoryGenerator` (any small script, or the existing `examples/basic_generation.py`). Pick one that should obviously prompt cooperation (allies/cooperative framing), one that should obviously prompt defection (enemies/competitive framing), and one fence-sitter. For each, also generate the formatted prompt that goes to the model (the same string the existing pipeline sends — including the payoff matrix and decision-format instructions).

Write them to `scripts/_smoke_stories.jsonl` as one JSON object per line:

```json
{"story_id": "smoke_coop_1", "prompt": "<full prompt text>", "coop_choice": "A"}
{"story_id": "smoke_def_1",  "prompt": "<full prompt text>", "coop_choice": "A"}
{"story_id": "smoke_neut_1", "prompt": "<full prompt text>", "coop_choice": "A"}
```

`coop_choice` is the letter ("A" or "B") that represents the cooperative play in the underlying payoff matrix.

- [ ] **Step 3: Run extraction on the smoke stories**

Run:
```bash
python3 scripts/run_steering.py extract \
    --run-id smoke1 \
    --stories scripts/_smoke_stories.jsonl \
    --split train
```

Expected output: a JSON summary like
```json
{
  "n_stories": 3,
  "decision_counts": {"A": 2, "B": 1},
  "coop_rate": 0.6666666666666666,
  "index_path": "/data/runs/smoke1/index.parquet"
}
```

If `decision_counts` shows mostly empty strings (`""`), the decision parser failed on the trace — inspect a bundle by downloading and printing `trace_text` (next step).

- [ ] **Step 4: Pull the bundles down for inspection**

Run:
```bash
modal volume get gtllm-steering runs/smoke1/ ./local_data/runs/smoke1/
```

Then:
```bash
python3 -c "
from pathlib import Path
from game_theory_llm.steering.storage import load_activation_bundle
for p in Path('./local_data/runs/smoke1/activations/train').glob('*.pt'):
    b = load_activation_bundle(p)
    print(b.story_id, b.decision, b.cooperated, len(b.activations), '<<', b.trace_text[:200])
"
```

Expected: three lines, each showing a story_id, A/B decision, True/False cooperated, the layer count (matching what `warmup` reported, e.g. 42), and the start of the trace text.

- [ ] **Step 5: Confirm one activation bundle has the right shape**

```bash
python3 -c "
from pathlib import Path
import torch
from game_theory_llm.steering.storage import load_activation_bundle
b = next(iter(Path('./local_data/runs/smoke1/activations/train').glob('*.pt')))
bundle = load_activation_bundle(b)
layer0 = bundle.activations[0]
for pos, t in layer0.items():
    print(pos, tuple(t.shape), t.dtype)
"
```

Expected: three lines, each showing `(<hidden_dim>,)` as the shape and `torch.bfloat16` as the dtype.

- [ ] **Step 6: Add a few more smoke stories so fitting will run**

Fitting requires `min_samples=30` per class by default. The smoke run uses three stories, so we won't fit on it. Skip the fit step for smoke; the real experiment (Task 12, future) will exercise fitting.

- [ ] **Step 7: Commit smoke verification**

```bash
git add .gitignore
git commit -m "chore(steering): gitignore local_data and smoke story files"
```

---

## After this plan: what comes next (not part of v1 implementation)

These are the operational steps to actually run an experiment, not code-writing tasks:

1. Generate ~250 PD stories with the existing `StoryGenerator`. Split: 200 train / 50 eval. Split by **topic**, not random — train and eval use disjoint topic sets to avoid topic-feature contamination.
2. Convert each story to the JSONL input format (`story_id`, `prompt`, `coop_choice`, `seed`).
3. Run `scripts/run_steering.py extract --split train` and `--split eval`.
4. Run `modal volume get gtllm-steering runs/<run_id>/ ./local_data/runs/<run_id>/`.
5. Run `scripts/run_steering.py fit`.
6. Run `scripts/run_steering.py push-vectors`.
7. Run `scripts/run_steering.py eval`.
8. Plot cooperation rate vs α per surviving (layer, position) cell.

The plotting + analysis layer is intentionally not in v1 — once you have `results_sweep.parquet`, the existing matplotlib/pandas tooling in the project can chart it in a notebook cell.
