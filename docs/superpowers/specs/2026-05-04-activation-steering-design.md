# Activation Steering for Cooperation in Gemma 4 E4B

**Date:** 2026-05-04
**Author:** Isaac Robinson (with brainstorming assistance)
**Status:** Design — pending implementation plan

## 1. Goal

Add activation-steering capability to the `game_theory_llm` library. Extract residual-stream activations from Gemma 4 E4B (instruction-tuned) on Prisoner's Dilemma stories, split by the model's natural decision (cooperate vs. defect), construct a per-(layer, token-position) steering vector via mean-difference, and validate by sweeping a steering coefficient α at inference and measuring cooperation-rate shift on held-out stories.

### 1.1 Scientific claim (v1)

> "On Gemma 4 E4B-it, there exists a residual-stream direction at layer L that, when added to activations at inference, monotonically shifts the model's cooperation rate on held-out Prisoner's Dilemma stories from baseline ~p₀ to ~p_high (at +α) and ~p_low (at −α)."

### 1.2 Out of scope for v1

- Multiple target models (E4B-it only)
- Framing-gap-closure analysis (deferred follow-up)
- Cross-topic / cross-actor generalization analysis
- Coherence / perplexity guardrails
- PCA / probe-based vector construction (mean-diff only)
- Multi-layer simultaneous steering
- Game types other than Prisoner's Dilemma

## 2. Target model

**`google/gemma-4-E4B-it`** (instruction-tuned variant of Gemma 4 E4B).

| Property | Value |
|---|---|
| Transformer layers | 42 |
| Effective parameters | 4.5B |
| Total parameters (incl. embeddings) | 8B |
| Vocabulary | 262K |
| Max context | 128K tokens |
| Native dtype | bf16 |
| License | Apache 2.0, not gated |
| Modalities | text, image, audio (we use text only) |
| Notable architecture | Per-Layer Embeddings (PLE), hybrid sliding-window + global attention with p-RoPE on global layers |

The vision (~150M) and audio (~300M) encoders are not loaded — we use the text-only causal-LM entry point.

The PLE mechanism gives each decoder layer its own small per-token embedding contribution. The residual stream is still the residual stream, so standard `register_forward_hook` on each transformer block should capture the layer output as it flows to the next block. The smoke test (Section 5.3) verifies this empirically before any real experiment runs.

## 3. Pipeline architecture

```
[Existing] StoryGenerator (OpenRouter) ──► stories.jsonl
                                              │
                                              ▼
                            ┌─────────────────────────────────┐
                            │ Modal: extract_activations()    │
                            │  load Gemma 4 E4B-it once       │
                            │  for each story:                │
                            │    sample a reasoning trace     │
                            │    register hooks on all 42     │
                            │      layers                     │
                            │    re-run prompt+trace forward  │
                            │      with hooks                 │
                            │    capture activations at:      │
                            │      pos1 = last prompt token   │
                            │      pos2 = last trace token    │
                            │      pos3 = mean over trace     │
                            │    parse decision from output   │
                            │    write ActivationBundle       │
                            └─────────────────────────────────┘
                                              │
                                              ▼
                                   activations/*.pt + index.parquet
                                              │
                                              ▼
                            ┌─────────────────────────────────┐
                            │ Local: fit_steering_vectors()   │
                            │  load activations index         │
                            │  split by decision (coop vs def)│
                            │  class-balance to smaller class │
                            │  for each (layer, position):    │
                            │    v = mean(coop) − mean(defect)│
                            │    raw_norm = ||v||             │
                            │    v ← v / raw_norm             │
                            │  → SteeringVectorSet            │
                            └─────────────────────────────────┘
                                              │
                                              ▼
                                       vectors.pt
                                              │
                                              ▼
                            ┌─────────────────────────────────┐
                            │ Modal: evaluate_steering()      │
                            │  PRUNE PASS:                    │
                            │   for each (layer, position):   │
                            │     eval at α = ±α_max only     │
                            │   score by coop_rate gap        │
                            │   keep top ~10 cells            │
                            │  SWEEP PASS:                    │
                            │   for each surviving cell:      │
                            │     eval full α grid            │
                            │   register add-hook at layer    │
                            │     h ← h + α·raw_norm·v        │
                            │   parse decisions               │
                            └─────────────────────────────────┘
                                              │
                                              ▼
                                      results.parquet
                                              │
                                              ▼
                                 plot: coop_rate vs α per layer
```

### 3.1 Extraction is two-pass

For each training story:
1. **Generation pass:** sample a reasoning trace from the model with the existing prompt format and a fixed temperature/seed.
2. **Hooked re-run:** concatenate prompt + sampled trace, run a single forward pass with hooks attached, capture activations at the three target token positions for all 42 layers.

This is simpler and more deterministic than capturing activations during generation, and the redundant compute is cheap relative to the eval stage.

### 3.2 Eval is two-pass (prune-then-sweep)

The full grid is 42 layers × 3 positions × 7 α values = 882 cells. Most cells will be uninteresting. We run:

1. **Pruning pass:** evaluate all 42 × 3 = 126 (layer, position) cells at just `+α_prune` and `−α_prune`, where `α_prune = max(α_grid) = 3`. Score each by `coop_rate(+α_prune) − coop_rate(−α_prune)`. Keep the top ~10 cells.
2. **Sweep pass:** evaluate the surviving cells across the full α grid `[-3, -2, -1, 0, +1, +2, +3]`.

All 126 cells' two-point scores are logged so we can spot-check whether anything got pruned suspiciously.

## 4. Data model

All dataclasses live in `game_theory_llm/steering/models.py`.

```python
@dataclass
class ActivationBundle:
    """Activations for a single story under a single model."""
    story_id: str
    model_name: str          # "gemma-4-e4b-it"
    decision: str            # parsed from the trace
    cooperated: bool         # decision == cooperative choice for this PD
    prompt_text: str
    trace_text: str
    # activations[layer_idx][position_key] -> 1D bf16 tensor of size hidden_dim
    # (stored in bf16 to halve disk usage; fitting upcasts to fp32 for arithmetic)
    activations: dict[int, dict[str, torch.Tensor]]
    # position_key in {"last_prompt", "last_trace", "mean_trace"}
    metadata: dict           # temperature, seed, hidden_dim, n_layers, etc.

@dataclass
class SteeringVector:
    """A single steering direction for one (layer, position)."""
    model_name: str
    layer: int
    position: str            # "last_prompt" | "last_trace" | "mean_trace"
    direction: torch.Tensor  # 1D fp32, hidden_dim, unit-normalized
    raw_norm: float          # ||mean_coop − mean_defect|| pre-normalization
    n_coop: int
    n_defect: int
    fit_metadata: dict       # corpus hash, fitting timestamp, etc.

@dataclass
class SteeringVectorSet:
    """Bundle of all (layer, position) vectors fit from one corpus."""
    model_name: str
    vectors: dict[tuple[int, str], SteeringVector]
    corpus_hash: str         # hash of (story_id, decision) pairs used for fitting

@dataclass
class SteeringEvalResult:
    """Cooperation rate for one (layer, position, α) cell on held-out stories."""
    layer: int
    position: str
    alpha: float
    n_stories: int
    n_cooperated: int
    cooperation_rate: float
    decisions: list[dict]    # per-story records: story_id, decision, raw output

@dataclass
class SteeringRun:
    """Full experiment manifest tying corpus → vectors → eval results."""
    run_id: str              # e.g. "2026-05-04-gemma-4-e4b-it-v1"
    model_name: str
    train_story_ids: list[str]
    eval_story_ids: list[str]
    vector_set_path: str
    results_path: str
    config: dict             # alpha grid, layers swept, positions, prune-threshold, etc.
```

`cooperated` is computed from `decision` plus the payoff matrix. The fitting split uses `cooperated`, not `decision`, so we are robust to label-letter ordering.

`corpus_hash` enables short-circuit caching of the fitting stage on repeated runs over the same activations.

### 4.1 Storage layout (Modal Volume)

```
/steering-data/
    runs/{run_id}/
        config.yaml
        activations/
            train/{story_id}.pt        # ActivationBundle, serialized
            eval/{story_id}.pt
        index.parquet                  # (story_id, decision, split, path)
        vectors.pt                     # SteeringVectorSet
        results.parquet                # one row per (layer, position, α, story_id)
```

Per-story `.pt` files (rather than one large tensor) make the extraction stage resumable: if a Modal worker dies mid-batch, we restart only the missing stories.

## 5. Modal setup

A single Modal app `gtllm-steering` defined in `game_theory_llm/steering/modal_app.py`.

### 5.1 Image and volume

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface_hub",
        "pandas", "numpy", "pyarrow",
    )
    .add_local_python_source("game_theory_llm")
)

volume = modal.Volume.from_name("gtllm-steering", create_if_missing=True)
app = modal.App("gtllm-steering", image=image)
```

No HuggingFace secret is needed — Gemma 4 E4B-it is Apache 2.0 and not gated.

### 5.2 Functions

```python
@app.cls(
    gpu="A100",                   # 40 GB; ~16 GB for weights + headroom
    volumes={"/data": volume},
    timeout=3600,
    scaledown_window=300,
)
class SteeringWorker:
    @modal.enter()
    def load(self):
        # Load Gemma 4 E4B-it once per container.
        # Use text-only causal-LM entry point (skip vision/audio encoders).

    @modal.method()
    def extract(self, stories: list[dict], run_id: str, split: str) -> dict: ...

    @modal.method()
    def evaluate(self, run_id: str, layers: list[int], alphas: list[float]) -> dict: ...

@app.function(gpu="A100", timeout=600)
def warmup() -> dict:
    """Smoke check. Returns n_layers, hidden_dim, module_path, dtype."""
```

The `@app.cls` pattern keeps the model in memory across calls within a container, avoiding the ~30s load cost per story.

### 5.3 GPU choice

Default: **A100 40 GB**. Reasons:

- Gemma 4 E4B's 8B total parameters at bf16 = ~16 GB of weights. T4 (16 GB total) has no headroom. L4 (24 GB) fits with margin. A100 40 GB has comfortable headroom for weights + activations + KV cache.
- Inference at batch 1 is memory-bandwidth-bound. A100's 1.5 TB/s vs L4's 300 GB/s gives roughly 3–4× wallclock speedup. A100 costs ~$3.10/hr on Modal vs L4's ~$1.25/hr — speedup roughly cancels price ratio, so total $$ is comparable but wallclock is much better.
- Iteration loop matters: eval going from "leave it for the afternoon" (L4) to "wait 20 minutes" (A100) changes how often we re-run.

L4 is a viable fallback by editing the `gpu=` argument if A100 capacity is constrained.

### 5.4 Cost estimate (v1)

| Stage | A100 wallclock | A100 cost |
|---|---|---|
| Extraction (~200 train stories, 2 forward passes each) | ~2 min | ~$0.10 |
| Eval pruning pass (126 cells × 50 stories × 2 α) | ~30 min | ~$1.50 |
| Eval sweep pass (10 cells × 50 stories × 7 α) | ~12 min | ~$0.60 |
| Buffer for iteration and re-runs | — | ~$10 |
| **Total** | **~1 hour active** | **~$10–25** |

### 5.5 Local driver

Orchestration lives in a normal Python script `scripts/run_steering.py` (not a `@app.local_entrypoint()`), which calls Modal functions over the network, parses existing `Story` JSONL files, dispatches batches, downloads results, and runs the local fitting stage. This keeps Modal code thin and orchestration debuggable without `modal run`.

## 6. Vector fitting and application

### 6.1 Fitting (`vector_fitting.py`)

For each `(layer, position)` cell:

```python
coop_acts   = stack([b.activations[layer][position] for b in bundles if b.cooperated])
defect_acts = stack([b.activations[layer][position] for b in bundles if not b.cooperated])

# Class-balance: subsample the larger class to match the smaller.
n = min(len(coop_acts), len(defect_acts))
coop_acts   = coop_acts[rng.choice(len(coop_acts),   n, replace=False)]
defect_acts = defect_acts[rng.choice(len(defect_acts), n, replace=False)]

# Cast to fp32 for the difference (extraction is bf16 for speed).
direction = coop_acts.float().mean(0) - defect_acts.float().mean(0)
raw_norm = direction.norm().item()
direction = direction / raw_norm
```

- **Minimum-sample threshold:** any cell where either class has fewer than 30 samples is skipped with a warning, not an error.
- **Stable seeds:** the subsampling RNG is seeded by `corpus_hash` so refitting is deterministic.
- **fp32 for the difference:** bf16 mean differences can lose precision when the means are close. Storage stays bf16; arithmetic upcasts.

### 6.2 Application (`application.py`)

```python
def make_steering_hook(direction: torch.Tensor, alpha: float, raw_norm: float):
    scaled = (alpha * raw_norm) * direction  # back to the natural scale of activations
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + scaled.to(h.device, h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

handle = model.model.layers[layer_idx].register_forward_hook(
    make_steering_hook(direction, alpha, raw_norm)
)
# ... generate ... then handle.remove()
```

### 6.3 Application choices (locked for v1)

1. **Apply at every token during generation.** Standard CAA — the hook fires on every forward pass, influencing every new token.
2. **Single layer at a time.** v1 adds the vector at the layer it was extracted from. Multi-layer addition is a v2 question.
3. **α grid:** `[-3, -2, -1, 0, +1, +2, +3]`. The α values are unitless multipliers applied to the raw (pre-normalization) inter-class gap. So α=1 means "shift activations by one inter-class gap." This makes results comparable across layers despite different natural activation scales. α=0 is the no-steering control.
4. **Generation config during eval:** identical temperature/seed schedule to the original story generation. The only thing changing between baseline and steered runs is the residual-stream addition.

The exact module path for hook attachment (`model.model.layers[i]` vs Gemma-specific path) is discovered by `warmup()` and read off the model object — not hard-coded.

## 7. Testing strategy

Two layers only. We deliberately avoid building a CI safety net for what is fundamentally a research pipeline driven by hand.

### 7.1 Unit tests (CPU, no Modal)

`tests/steering/test_vector_fitting.py`:
- Synthetic activations with known cooperate/defect means → fitted direction matches expected.
- Class-balancing subsamples to the smaller class.
- All-cooperate or all-defect input raises a clear error.

~3 tests, ~50 lines.

### 7.2 Modal smoke check

The `warmup()` function. Run it once before any real experiment. Returns `(n_layers, hidden_dim, module_path, dtype)` and the result of attaching a no-op forward hook to verify hook-attachment works on Gemma 4. Costs pennies.

### 7.3 Things deliberately not tested

- "Does the steering vector actually steer cooperation?" — that is the experiment, not a test.
- Generated-text snapshots — sampling is non-deterministic enough to be brittle, and seeded determinism on GPU is unreliable across CUDA versions.

## 8. Risks and failure modes

In rough order of probability.

### 8.1 PLE behavior under hooks is unverified

Gemma 4's Per-Layer Embeddings add a per-token, per-layer embedding contribution inside each decoder layer. We need to verify that `register_forward_hook` on `model.model.layers[i]` captures the residual stream **after** the PLE contribution (which is what flows to the next layer and what we want to steer). **Mitigation:** the smoke check runs a hook on a real forward pass and verifies the captured tensor matches the next-layer input. Discovered before any real experiment.

### 8.2 Decision split is too skewed to fit a vector

If Gemma 4 E4B-it cooperates ≥95% (or ≤5%) of the time on the existing corpus, the minority class is too small for a clean mean. **Mitigation:** before extraction, check the decision distribution on a quick sample. If skewed, generate additional stories with framings (`actor_type="enemies"`, competitive topics) known to push the model the other way until each class has ≥50 examples.

### 8.3 Steering destroys output coherence at useful α

The vector might shift cooperation rate only at α large enough to wreck the text. v1 explicitly skips the coherence guardrail. **Mitigation:** spot-check a handful of generated traces by eye at the chosen α before claiming results. If they are word salad, escalate to a v2 that adds a perplexity check.

### 8.4 Train / eval contamination via topic

If we train and eval on stories from the same `(topic, actor_type, world_type)` combos, the vector might pick up topic features rather than cooperation features. **Mitigation:** split by **topic**, not by random sampling — train on a subset of topics, eval on disjoint topics.

### 8.5 Vision/audio encoders inflate memory or get loaded by accident

Loading the multimodal classes adds ~450M extra params and may change module paths. **Mitigation:** load via `AutoModelForCausalLM` with the text-only entry point. The smoke check verifies the loaded model has only the text decoder.

## 9. Module layout

```
game_theory_llm/steering/
    __init__.py
    modal_app.py          # Modal app: image, volumes, GPU spec, SteeringWorker class
    extraction.py         # forward pass with hooks → ActivationBundle
    vector_fitting.py     # mean-diff per (layer, position) → SteeringVector
    application.py        # hook factory; generate-with-steering helper
    evaluation.py         # prune-then-sweep eval over (layer, position, α)
    storage.py            # serialization (per-story .pt + Parquet index)
    models.py             # ActivationBundle, SteeringVector, SteeringVectorSet,
                          #   SteeringEvalResult, SteeringRun

tests/steering/
    test_vector_fitting.py

scripts/
    run_steering.py       # local driver: orchestrates extract → fit → evaluate

docs/superpowers/specs/
    2026-05-04-activation-steering-design.md  (this file)
```

`torch` and `transformers` go under a new `[steering]` extra in `pyproject.toml` so users doing only black-box analysis are not forced to install heavy CUDA-adjacent dependencies.

## 10. Deferred follow-ups (not v1, but we want the architecture to admit them)

| Follow-up | What it adds | Why deferred |
|---|---|---|
| Framing-gap-closure analysis | Does +α·v on enemy-framed stories reproduce the cooperation rate of ally-framed stories? | Separate analysis on top of the same vectors; not on the critical path for v1. |
| Coherence/perplexity guardrails | Held-out judge or reference-model perplexity to verify steered text stays coherent. | v1 covers via spot-check. |
| Cross-topic / cross-actor generalization | Train on T1, eval on T2. | Cheap to add once v1 works; not on the critical path for the first claim. |
| Multi-layer steering | Add v at multiple layers simultaneously. | More design choices; v1 establishes single-layer baseline first. |
| PCA / probe-based vector construction | Alternative to mean-diff. | Mean-diff is the standard CAA baseline; alternatives are interesting follow-ups. |
| Other open-weight model targets (Llama, Qwen, reasoning-model variants) | Generalization claim. | Belongs in a v2 once the pipeline is solid on Gemma 4 E4B-it. |
