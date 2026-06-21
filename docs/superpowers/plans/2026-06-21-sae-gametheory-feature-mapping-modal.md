# SAE Feature-Mapping of Game-Theoretic Reasoning — Implementation Plan (Modal v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the cooperate↔defect direction and a game-theoretic "recognition" representation into interpretable Qwen-Scope SAE features on `Qwen3.5-9B-Base`, then causally test them (clamp → P(coop) dose-response; ablate recognition → mediation). **The science is unchanged from the local plan / approved spec.** The only change is *where the model runs*: the heavy 9B forward passes are executed on Modal (A100-80GB) by reusing this repo's existing steering Modal infrastructure; the SAE math and all discovery/interpretation/analysis stay local.

**Architecture (the split):** **heavy 9B forward passes run on Modal; the SAE math + all discovery/analysis stay LOCAL.**
- A new Modal worker `SaemapWorker` (a new `@app.cls` on the SAME `safety` app — do NOT rename the app/volume) is **MODEL-ONLY: it does NOT load the SAE.** It exposes three methods: `extract_residuals` (residuals at chosen layers, optionally pooled over a completion span), `pcoop` (next-token P(coop) over {A,B}), and `causal_pcoop` (the same readout with a residual-ADD steering hook applied during the forward pass).
- The controller holds the SAE weights locally (downloaded in Task 1 to `paths.SAE_CACHE`) and `sae.py`. It computes decoder columns / SAE encodings locally; for causal steering it sends the worker a plain 4096-d vector to add at a layer. The residual-ADD hook on the worker IS feature-clamping (add `α·unit(W_dec[:,f])`); ablation is just a negative/scaled vector the controller supplies.
- Big tensors do not cross the boundary as return values where avoidable. Small residual matrices (`[N, n_layers, 4096]` bf16, tens of MB) are returned directly; if `N` is large the worker torch-saves the tensor to the `safety` volume and returns a path (following the existing `_impl_extract` write-to-volume + `volume.commit()` pattern).
- A thin local client `game_theory_llm/saemap/remote.py` invokes the worker via `modal.Cls.from_name("safety", "SaemapWorker")` (mirroring `scripts/run_full_pipeline.py:125`) and exposes local functions `extract_residuals(...)`, `pcoop(...)`, `causal_pcoop(...)` returning numpy/torch. **This `remote.py` REPLACES the local MPS `model.py` of the original Task 3.** Every later script imports `from game_theory_llm.saemap import remote` instead of `QwenModel`.

**Tech Stack:** Local: Python 3.11 venv (`.venv-sae`) with `torch` (CPU is fine — the SAE math is small), `huggingface_hub`, `safetensors`, `scikit-learn`, `numpy`, `pandas`, `scipy`, `pytest`, and the `modal` client. Remote: the `safety` Modal app, A100-80GB, a **new `saemap_image`** based on the existing steering image but pinned to `transformers>=5.10` (Qwen3.5 support; see Global Constraints / Task 3).

## Global Constraints

- **Spec:** `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/docs/superpowers/specs/2026-06-18-sae-cooperation-feature-mapping-design.md` — the experiment (decision + recognition tracks, mediation, controls, Step-0 gate) is unchanged.
- **Work in the worktree:** all code/commits on branch `feature/activation-steering` at `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/` (cwd for every command below).
- **Local interpreter:** use `.venv-sae/bin/python` for all SAE/discovery/analysis code (Python 3.11) **and** to drive Modal (`pip install modal` into `.venv-sae`). Do NOT use system `python3` (3.9). Pure-math unit tests also run under `.venv-sae/bin/python -m pytest`.
- **Modal app + volume:** `safety` (project rule: Modal resources are named `safety`; do NOT create or rename to anything else). **GPU:** `A100-80GB`. The model lives on the `safety` volume at `/data/models/saemap_9b` (= `model_local_path("saemap_9b")`). The SAE weight files stay LOCAL under `paths.SAE_CACHE`; **do NOT push the SAE to the volume.**
- **Image:** the worker is decorated with a NEW `saemap_image` (defined in `modal_app.py`) — the existing steering `image` pins `transformers>=4.46` which loads Gemma but NOT Qwen3.5; `saemap_image` is the same base with `transformers>=5.10`. Do NOT modify the existing `image` (Gemma's `SteeringWorker`/`SteeringWorkerLarge` need 4.46).
- **Model:** `Qwen/Qwen3.5-9B-Base`. **SAE repo:** `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` (per-layer `.pt`: keys `W_enc` (65536,4096), `W_dec` (4096,65536), `b_enc` (65536,), `b_dec` (4096,); d_model 4096, d_sae 65536, **TopK=50 applied at runtime, no activation fn**).
- **SAE math (LOCAL, unchanged):** `pre = resid @ W_enc.T + b_enc`; keep top-50 per token (zero the rest); `recon = acts @ W_dec.T + b_dec`; feature `f` decoder direction = `W_dec[:, f]`.
- **Layer indexing (unchanged):** SAE "layer L" = output of transformer block L = `model.model.layers[L]` (the module the worker's `_discover_layers` returns). The worker captures the residual via a forward hook on that block (the existing `extraction.py` hook approach), NOT `output_hidden_states` (avoids the off-by-one index ambiguity). Validated empirically by the reconstruction-fidelity check (Task 3 Step 6): `QwenScopeSAE.variance_explained` must be > 0.5 at the chosen layer.
- **Candidate layers:** `[8, 12, 16, 20, 24]`. **Memory rule (LOCAL):** never hold more than one SAE layer in memory at once (load per-layer); the model is on the GPU remotely, so local peak is one SAE (~1.1 GB) plus returned residual matrices (tens of MB).
- **Decision readout (unchanged):** PD prompts already end with `...your decision, either: <decision>A</decision> or <decision>B</decision>...`. P(coop) = next-token prob of the cooperative letter after appending `\n<decision>`, normalized over {A,B}; `coop_choice` gives the cooperative letter; optional few-shot prefix toggled by the Step-0 gate.
- **No regex for quality/semantic judgments** (project rule); feature *labels* come from max-activating examples, optionally an LLM judge — never regex pattern-matching to claim a feature's meaning.
- **Run dir (LOCAL):** `data/runs/saemap_9b/` (under worktree). The worker's volume run dir (for large saved tensors) is `/data/runs/saemap_9b/` on the `safety` volume.

---

## File Structure

```
game_theory_llm/saemap/
    __init__.py        # exports (already implemented)
    paths.py           # absolute paths to corpora (main + worktree), ids, layer list, run dir (already implemented)
    sae.py             # QwenScopeSAE: load/encode(topk)/decoder_col/reconstruct/variance_explained (already implemented, LOCAL, unchanged)
    remote.py          # NEW — thin local client over Modal SaemapWorker: extract_residuals(), pcoop(), causal_pcoop()
    corpus.py          # decision cue-pairs (+swap); recognition class sets (dilemma/nondilemma/game/nongame)
    discover.py        # diff_of_means; l1_probe; decompose_direction; held_out_auc
    interpret.py       # max_activating_examples (uses remote.extract_residuals + local sae.encode)
    causal.py          # decision_sweep; mediation; random_feature_control (all via remote.causal_pcoop / remote.pcoop)
game_theory_llm/steering/
    modal_app.py       # MODIFY — add saemap_image, download_saemap_model(), and the SaemapWorker @app.cls
scripts/
    saemap_setup_env.sh        # create .venv-sae + install deps (incl. modal) + download SAE layers locally (already implemented; +modal)
    saemap_push_model.py       # NEW — invoke download_saemap_model on the safety volume
    saemap_sanity.py           # Step-0 HARD GATE: P(coop) non-degenerate + responsive (via remote.pcoop)
    saemap_extract.py          # cache decision cue-span + recognition scenario-span SAE features per layer (via remote.extract_residuals + local sae.encode)
    saemap_discover.py         # run A/B/C per contrast -> discover/*.json + interpret top features
    saemap_causal.py           # decision sweep + mediation + controls -> causal/*.json + verdict (via remote.causal_pcoop)
tests/saemap/
    __init__.py
    test_sae.py          # synthetic: encode shape, L0==50, reconstruct round-trip (already implemented)
    test_corpus.py       # cue-pair + label-swap + class-set membership
    test_discover.py     # diff_of_means + l1_probe recover a planted signal
    test_causal_hook.py  # local arithmetic of decision_sweep/mediation against a fake remote (no GPU)
```

---

## Task 1: Environment + paths module + SAE weight download (LOCAL)

> **Already implemented** — `game_theory_llm/saemap/__init__.py`, `game_theory_llm/saemap/paths.py`, `scripts/saemap_setup_env.sh`, and the `data/runs/saemap_9b/` run dirs exist in the worktree. Copy this task verbatim; the only Modal-v2 addition is **also `pip install modal` into `.venv-sae`** so the controller can drive the Modal app (Step 3 note below). The local model snapshot pre-fetch is no longer required (the model is loaded remotely from the `safety` volume — see Task 3), but harmless if left in.

**Files:**
- Create: `scripts/saemap_setup_env.sh`
- Create: `game_theory_llm/saemap/__init__.py`
- Create: `game_theory_llm/saemap/paths.py`
- Test: `scripts/saemap_setup_env.sh` self-checks (no pytest — this is env/IO)

**Interfaces:**
- Produces: importable `game_theory_llm.saemap.paths` with constants used by every later task:
  `MODEL_ID, SAE_REPO, D_MODEL=4096, D_SAE=65536, TOPK=50, CANDIDATE_LAYERS, WORKTREE, MAIN_REPO, RUN_DIR, SAE_CACHE, PD_TRAIN, PD_EVAL, PD_SWAP, STORIES_DIR, NONGAME_FILES, DILEMMA_GAMES, NONDILEMMA_GAMES, ALL_GAMES`.

- [ ] **Step 1: Write `game_theory_llm/saemap/paths.py`**

```python
"""Absolute paths + constants for the SAE feature-mapping project.

Data is split across two checkouts: per-game story files live in the MAIN repo,
the processed PD steering corpus + non-game corpora live in the worktree.
All paths are absolute so scripts work from any cwd.
"""
from pathlib import Path

MAIN_REPO = Path("/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations")
WORKTREE = MAIN_REPO / ".worktrees/steering"

MODEL_ID = "Qwen/Qwen3.5-9B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
D_MODEL = 4096
D_SAE = 65536
TOPK = 50
CANDIDATE_LAYERS = [8, 12, 16, 20, 24]

RUN_DIR = WORKTREE / "data/runs/saemap_9b"
SAE_CACHE = RUN_DIR / "sae_cache"            # downloaded layer{L}.sae.pt files

# Decision track (worktree)
_STEER = WORKTREE / "data/runs/2026-05-05-sharp/steering"
PD_TRAIN = _STEER / "train.jsonl"
PD_EVAL = _STEER / "eval.jsonl"
PD_SWAP = _STEER / "full_corpus_swap.jsonl"

# Recognition track — per-game stories (MAIN repo; key = "content")
STORIES_DIR = MAIN_REPO / "data/runs/2026-05-05-sharp/stories"
DILEMMA_GAMES = ["prisoners_dilemma", "stag_hunt", "chicken"]
NONDILEMMA_GAMES = ["harmony", "deadlock"]
ALL_GAMES = DILEMMA_GAMES + NONDILEMMA_GAMES + ["battle_of_the_sexes", "matching_pennies"]

# Recognition coarse negative — non-game corpora (worktree; key = "prompt")
NONGAME_FILES = [
    WORKTREE / "data/runs/bbh/logical_deduction_eval.jsonl",
    WORKTREE / "data/runs/ethics_deontology/eval_subset100.jsonl",
    WORKTREE / "data/runs/ethics_util/eval_powered300.jsonl",
    WORKTREE / "data/runs/capability/gsm8k_eval.jsonl",
]


def ensure_run_dirs() -> None:
    for p in [RUN_DIR, SAE_CACHE, RUN_DIR / "activations",
              RUN_DIR / "discover", RUN_DIR / "causal", RUN_DIR / "interpret"]:
        p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Write `game_theory_llm/saemap/__init__.py`**

```python
from . import paths  # noqa: F401
```

- [ ] **Step 3: Write `scripts/saemap_setup_env.sh`**

```bash
#!/usr/bin/env bash
# Create the Python 3.11 venv for SAE work and download candidate SAE layers locally.
# The 9B model itself is NOT downloaded locally in Modal v2 — it is loaded remotely
# from the safety volume (see scripts/saemap_push_model.py). We DO install the modal
# client so the controller can drive the safety app.
set -euo pipefail
cd "$(dirname "$0")/.."          # worktree root

if [ ! -d .venv-sae ]; then
  python3.11 -m venv .venv-sae
fi
.venv-sae/bin/python -m pip install -q --upgrade pip
.venv-sae/bin/python -m pip install -q \
  "torch>=2.4" "huggingface_hub>=0.36" safetensors \
  "scikit-learn>=1.4" numpy pandas scipy pytest "modal>=0.64"

# Download candidate SAE layer files into the run cache (per-layer, ~1GB each, LOCAL).
.venv-sae/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from game_theory_llm.saemap import paths
paths.ensure_run_dirs()
for L in paths.CANDIDATE_LAYERS:
    fn = f"layer{L}.sae.pt"
    p = hf_hub_download(repo_id=paths.SAE_REPO, filename=fn,
                        local_dir=str(paths.SAE_CACHE))
    print("downloaded", p)
PY

# Confirm the modal client is configured (token present); does not run anything heavy.
.venv-sae/bin/python -c "import modal; print('OK modal', modal.__version__)"
echo "ENV READY"
```

- [ ] **Step 4: Run env setup**

Run: `bash scripts/saemap_setup_env.sh`
Expected: five `downloaded .../layer{L}.sae.pt`, `OK modal <version>`, `ENV READY`. If the SAE filename differs, list the repo: `.venv-sae/bin/python -c "from huggingface_hub import list_repo_files; print([f for f in list_repo_files('Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50') if 'layer' in f][:5])"` and adjust the `fn` pattern. If `import modal` fails on auth, run `.venv-sae/bin/python -m modal setup` once.

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/saemap/__init__.py game_theory_llm/saemap/paths.py scripts/saemap_setup_env.sh
git commit -m "saemap(modal): env setup, paths module, local SAE weight download + modal client"
```

---

## Task 2: SAE loader (`sae.py`) — pure math + reconstruction fidelity (LOCAL)

> **Already implemented** — `game_theory_llm/saemap/sae.py`, `tests/saemap/__init__.py`, and `tests/saemap/test_sae.py` exist in the worktree. This unit is local and device-agnostic; it is unchanged in Modal v2 (the recon-fidelity *run* moves to Task 3 Step 6, where residuals come back from the worker). Copy this task verbatim.

**Files:**
- Create: `game_theory_llm/saemap/sae.py`
- Test: `tests/saemap/test_sae.py`, `tests/saemap/__init__.py`

**Interfaces:**
- Consumes: `paths.SAE_CACHE`, `paths.TOPK`, `paths.D_MODEL`, `paths.D_SAE`.
- Produces: `class QwenScopeSAE` with:
  - `QwenScopeSAE.load(layer:int, device="cpu", dtype=torch.float32) -> QwenScopeSAE`
  - attrs `W_enc [D_SAE,D_MODEL]`, `W_dec [D_MODEL,D_SAE]`, `b_enc [D_SAE]`, `b_dec [D_MODEL]`, `k:int`, `layer:int`
  - `encode(resid: Tensor[...,D_MODEL]) -> Tensor[...,D_SAE]` (top-k sparse, rest zero)
  - `decoder_col(f:int, unit:bool=False) -> Tensor[D_MODEL]`
  - `reconstruct(acts: Tensor[...,D_SAE]) -> Tensor[...,D_MODEL]`
  - `variance_explained(resid: Tensor[N,D_MODEL]) -> float`

- [ ] **Step 1: Write the failing test** (`tests/saemap/__init__.py` empty; `tests/saemap/test_sae.py`)

```python
import torch
from game_theory_llm.saemap.sae import QwenScopeSAE

def _toy_sae(d_model=8, d_sae=32, k=4):
    sae = QwenScopeSAE.__new__(QwenScopeSAE)
    torch.manual_seed(0)
    sae.W_enc = torch.randn(d_sae, d_model)
    sae.W_dec = torch.randn(d_model, d_sae)
    sae.b_enc = torch.zeros(d_sae)
    sae.b_dec = torch.zeros(d_model)
    sae.k = k
    sae.layer = 0
    return sae

def test_encode_is_topk_sparse():
    sae = _toy_sae()
    resid = torch.randn(5, 8)
    acts = sae.encode(resid)
    assert acts.shape == (5, 32)
    # exactly k non-zero per row
    assert (acts != 0).sum(dim=-1).tolist() == [4, 4, 4, 4, 4]

def test_decoder_col_unit_norm():
    sae = _toy_sae()
    v = sae.decoder_col(3, unit=True)
    assert v.shape == (8,)
    assert abs(v.norm().item() - 1.0) < 1e-5

def test_reconstruct_shape():
    sae = _toy_sae()
    acts = sae.encode(torch.randn(5, 8))
    rec = sae.reconstruct(acts)
    assert rec.shape == (5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_sae.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `QwenScopeSAE`.

- [ ] **Step 3: Write `game_theory_llm/saemap/sae.py`**

```python
"""Qwen-Scope SAE loader (TopK, residual-stream)."""
from __future__ import annotations
import torch
from .paths import SAE_CACHE, TOPK

class QwenScopeSAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, layer, k=TOPK):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.layer, self.k = layer, k

    @classmethod
    def load(cls, layer: int, device="cpu", dtype=torch.float32) -> "QwenScopeSAE":
        sd = torch.load(SAE_CACHE / f"layer{layer}.sae.pt", map_location="cpu")
        g = lambda key: sd[key].to(device=device, dtype=dtype)
        return cls(g("W_enc"), g("W_dec"), g("b_enc"), g("b_dec"), layer)

    def encode(self, resid: torch.Tensor) -> torch.Tensor:
        pre = resid @ self.W_enc.T + self.b_enc           # [..., D_SAE]
        topv, topi = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        return acts

    def decoder_col(self, f: int, unit: bool = False) -> torch.Tensor:
        v = self.W_dec[:, f]
        return v / v.norm() if unit else v

    def reconstruct(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec.T + self.b_dec

    def variance_explained(self, resid: torch.Tensor) -> float:
        rec = self.reconstruct(self.encode(resid))
        num = (resid - rec).pow(2).sum().item()
        den = (resid - resid.mean(0, keepdim=True)).pow(2).sum().item()
        return 1.0 - num / den
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_sae.py -v`
Expected: 3 passed.

- [ ] **Step 5: Reconstruction-fidelity smoke (real layer; validates orientation + layer index)**

This requires the Modal worker (Task 3). **Defer the run to Task 3 Step 6** — there the recon check pulls real residuals back from `SaemapWorker.extract_residuals` and asserts `QwenScopeSAE.variance_explained > 0.5`. For this task, just confirm the math units pass.

- [ ] **Step 6: Commit**

```bash
git add game_theory_llm/saemap/sae.py tests/saemap/__init__.py tests/saemap/test_sae.py
git commit -m "saemap(modal): Qwen-Scope SAE loader (topk encode, reconstruct) + tests"
```

---

## Task 3: Modal worker (`SaemapWorker`) + `saemap_image` + push model to volume + local client (`remote.py`) + recon fidelity

This is the central Modal-v2 task. It **replaces the local MPS `model.py`** of the original plan. We (a) add a new `saemap_image` and the `SaemapWorker` `@app.cls` to the existing `modal_app.py` (reusing `_worker_load_impl`, `_discover_layers`, and the `make_steering_hook` residual-add); (b) push the Qwen3.5-9B-Base weights to the `safety` volume; (c) write the thin local client `remote.py`; (d) run the reconstruction-fidelity check remotely.

**Files:**
- Modify: `game_theory_llm/steering/modal_app.py` (add `saemap_image`, `download_saemap_model`, `SaemapWorker`)
- Create: `scripts/saemap_push_model.py`
- Create: `game_theory_llm/saemap/remote.py`
- Test: integration smoke commands (documented inline; no mocked unit test — worker code is GPU-bound)

**Interfaces:**
- `SaemapWorker` (`@app.cls`, `gpu="A100-80GB"`, `image=saemap_image`, `volumes={"/data": volume}`), `model_name: str = modal.parameter(default="saemap_9b")`, `@modal.enter()` `load()` calls `_worker_load_impl(self, self.model_name)` then resolves A/B token ids. Three `@modal.method()`s:
  - `extract_residuals(self, prompts: list[str], layers: list[int], completion: str | None = None) -> list[list[list[float]]]` → an `[N, len(layers), 4096]` nested list (bf16→float32 on the way out). One forward pass per prompt; capture residuals at each requested layer's block output via a forward hook; if `completion` is given, append it to the prompt and mean-pool the residual over the completion-span tokens, else mean-pool over all prompt tokens.
  - `pcoop(self, prompts: list[str], coop_letters: list[str], fewshot: str = "") -> list[float]` → for each prompt, append `fewshot + prompt + "\n<decision>"`, read next-token logits, softmax over the two resolved {A,B} token-ids, return the prob of `coop_letters[i]`.
  - `causal_pcoop(self, prompts: list[str], coop_letters: list[str], layer: int, vec: list[float], fewshot: str = "") -> list[float]` → same readout as `pcoop` but with a residual-ADD hook (`make_steering_hook(unit_dir, alpha, raw_norm=1.0)` form) adding the 4096-d `vec` at `self.layers[layer]` during each forward pass.
- `remote.py` (LOCAL): `extract_residuals(prompts, layers, completion=None) -> np.ndarray [N, len(layers), 4096]`; `pcoop(prompts, coop_letters, fewshot="") -> np.ndarray [N]`; `causal_pcoop(prompts, coop_letters, layer, vec, fewshot="") -> np.ndarray [N]`. Each builds the worker via `modal.Cls.from_name("safety", "SaemapWorker")(model_name="saemap_9b")` and calls `.<method>.remote(...)`. A module-level `_worker()` memoizes the handle.

- [ ] **Step 1: Add `saemap_image` + `download_saemap_model` to `game_theory_llm/steering/modal_app.py`**

Append AFTER the existing `image`/`volume`/`app` definitions (do NOT touch the existing `image`). The worker reuses the SAME `app` and `volume`.

```python
# --- SAE feature-mapping (saemap) additions -------------------------------
# Qwen3.5-9B-Base does NOT load under transformers>=4.46 (the existing `image`
# pin, required by Gemma). It needs transformers ~5.x (config nests the decoder
# under text_config). Define a SEPARATE image so we never regress the Gemma
# workers. Same base + deps; only the transformers pin differs.
saemap_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=5.10",          # Qwen3.5 support (local validated 5.12.1)
        "accelerate>=1.0",
        "huggingface_hub",
        "safetensors",
        "numpy>=1.24",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("game_theory_llm")
)

SAEMAP_MODEL_NAME = "saemap_9b"          # volume dir = /data/models/saemap_9b
SAEMAP_HF_ID = "Qwen/Qwen3.5-9B-Base"


@app.function(image=saemap_image, volumes={"/data": volume}, timeout=3600)
def download_saemap_model(hf_id: str = SAEMAP_HF_ID,
                          local_name: str = SAEMAP_MODEL_NAME) -> dict:
    """Snapshot Qwen3.5-9B-Base into the safety volume at /data/models/<local_name>.

    Mirrors download_model_once but writes under a stable local_name so the worker
    loads from /data/models/saemap_9b regardless of the HF repo id. Idempotent.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(model_local_path(local_name))   # /data/models/saemap_9b
    if (target / "config.json").exists():
        n_files = sum(1 for _ in target.rglob("*"))
        return {"already_present": True, "hf_id": hf_id,
                "path": str(target), "n_files": n_files}
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download_saemap] {hf_id} -> {target}", flush=True)
    snapshot_download(repo_id=hf_id, local_dir=str(target),
                      local_dir_use_symlinks=False)
    volume.commit()
    n_files = sum(1 for _ in target.rglob("*"))
    print(f"[download_saemap] done; {n_files} files", flush=True)
    return {"downloaded": True, "hf_id": hf_id, "path": str(target), "n_files": n_files}
```

Note: `_worker_load_impl(self, model_name)` calls `model_local_path(model_name)` → `/data/models/saemap_9b` when `model_name="saemap_9b"`, so the worker loads the volume copy with no further change. `_discover_layers` tries `model.model.layers` (its 4th candidate path) — for Qwen3.5-9B-Base (non-multimodal) this is the decoder ModuleList; the implementer confirms `len(self.layers) == 32` from the `load()` log line and the Step-6 recon check validates the layer→SAE mapping.

- [ ] **Step 2: Add the `SaemapWorker` `@app.cls` to `game_theory_llm/steering/modal_app.py`**

Append after `SaemapWorker`'s image/download helpers (e.g. after `SteeringWorkerLarge`). It reuses `_worker_load_impl`, `_discover_layers`, and `make_steering_hook`.

```python
@app.cls(
    gpu="A100-80GB",
    image=saemap_image,
    volumes={"/data": volume},
    timeout=43200,
    scaledown_window=300,
)
class SaemapWorker:
    """A100-80GB MODEL-ONLY worker for Qwen3.5-9B-Base SAE feature-mapping.

    Loads ONLY the model (never the SAE — the controller holds the SAE locally).
    Captures residuals via forward hooks on self.layers[L] (same approach as
    steering/extraction.py), reads P(coop) from next-token logits over {A,B},
    and applies a residual-ADD steering hook for causal P(coop).
    """
    model_name: str = modal.parameter(default=SAEMAP_MODEL_NAME)

    @modal.enter()
    def load(self):
        _worker_load_impl(self, self.model_name)   # sets self.model/tokenizer/layers
        # Resolve the bare {A,B} token ids once (first sub-token as it follows
        # "<decision>"). add_special_tokens=False so we get the raw letter token.
        self.tid_A = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        self.tid_B = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        print(f"[saemap] A/B token ids = {self.tid_A}/{self.tid_B}", flush=True)

    # --- internal: one hooked forward, residuals at requested layers ---------
    def _residuals_one(self, text: str, layers: list, span: slice):
        """Return [len(layers), D_MODEL] float32: residual mean-pooled over `span`."""
        import torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        captured = {}

        def make_hook(L):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[L] = h[0].detach()       # [T, D_MODEL] on device
                return out
            return hook

        handles = [self.layers[L].register_forward_hook(make_hook(L)) for L in layers]
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for h in handles:
                h.remove()
        rows = []
        for L in layers:
            h = captured[L]                       # [T, D_MODEL]
            rows.append(h[span].float().mean(0).cpu())
        return torch.stack(rows)                  # [len(layers), D_MODEL]

    @modal.method()
    def extract_residuals(self, prompts: list[str], layers: list[int],
                          completion: str | None = None) -> list:
        """One forward per prompt; residual mean-pooled over the completion span
        (if `completion` given) else over all prompt tokens. Returns a nested list
        [N, len(layers), D_MODEL] (float32)."""
        import torch
        out = []
        for p in prompts:
            if completion is not None:
                pid = self.tokenizer(p, return_tensors="pt").input_ids
                start = pid.shape[1]
                text = p + completion
                full = self.tokenizer(text, return_tensors="pt").input_ids
                span = slice(start, full.shape[1])
            else:
                pid = self.tokenizer(p, return_tensors="pt").input_ids
                text = p
                span = slice(0, pid.shape[1])
            r = self._residuals_one(text, layers, span)   # [len(layers), D_MODEL]
            out.append(r.tolist())
        return out

    # --- internal: P(coop) for one text under whatever hooks are active ------
    def _pcoop_text(self, text: str, coop_letter: str) -> float:
        import torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        a = logits[self.tid_A].item()
        b = logits[self.tid_B].item()
        pa, pb = torch.softmax(torch.tensor([a, b]), 0).tolist()
        return pa if coop_letter == "A" else pb

    @modal.method()
    def pcoop(self, prompts: list[str], coop_letters: list[str],
              fewshot: str = "") -> list:
        """Normalized P(coop) over {A,B} after appending '\\n<decision>'."""
        return [self._pcoop_text(fewshot + p + "\n<decision>", c)
                for p, c in zip(prompts, coop_letters)]

    @modal.method()
    def causal_pcoop(self, prompts: list[str], coop_letters: list[str],
                     layer: int, vec: list[float], fewshot: str = "") -> list:
        """Same readout as pcoop but ADD the 4096-d `vec` to self.layers[layer]'s
        residual on every forward (reuses steering's residual-add hook)."""
        import torch
        from game_theory_llm.steering.application import make_steering_hook
        v = torch.tensor(vec, dtype=torch.float32)
        hook = make_steering_hook(v, alpha=1.0, raw_norm=1.0)   # adds 1*1*v
        handle = self.layers[layer].register_forward_hook(hook)
        try:
            return [self._pcoop_text(fewshot + p + "\n<decision>", c)
                    for p, c in zip(prompts, coop_letters)]
        finally:
            handle.remove()
```

Notes on reuse:
- `make_steering_hook(direction, alpha, raw_norm)` (application.py:18) adds `(alpha*raw_norm)*direction` to the layer output's residual at every token position and correctly handles the tuple-output case. Passing `alpha=1.0, raw_norm=1.0` makes it a plain `+vec` add — exactly the feature-clamp/ablation primitive this architecture needs. The controller scales `vec` (`α·unit(W_dec[:,f])`, or its negation for ablation), so the worker never needs the SAE.
- The residual capture mirrors `extraction.py`'s `make_hook` (block output `out[0] if tuple`).

- [ ] **Step 2b: Deploy the updated `safety` app** (REQUIRED before any `from_name` call)

`modal.Cls.from_name("safety", "SaemapWorker")` and `modal.Function.from_name("safety", "download_saemap_model")` resolve a **deployed** app, so the app must be (re)deployed after adding the worker. `modal` is already installed in `.venv-sae`; Modal auth is configured (`~/.modal.toml`). Deploying rebuilds only the new `saemap_image` (transformers 5.10, ~a few minutes); the existing `image` and Gemma workers are unchanged/cached.

```bash
.venv-sae/bin/python -m modal deploy game_theory_llm/steering/modal_app.py
```
Expected: a successful deploy listing the `safety` app objects including `SaemapWorker` and `download_saemap_model`. If the `saemap_image` build fails on the transformers pin, fall back to `transformers==5.12.1` (the locally-validated version) in `saemap_image` and re-deploy. Re-deploy after ANY later change to `modal_app.py`.

- [ ] **Step 3: Write `scripts/saemap_push_model.py` and push the model to the volume**

```python
"""Push Qwen3.5-9B-Base onto the safety volume at /data/models/saemap_9b."""
import json
import modal
from game_theory_llm.steering import modal_app  # noqa: F401 ensure app is importable

def main():
    fn = modal.Function.from_name("safety", "download_saemap_model")
    print("[push] downloading Qwen3.5-9B-Base into safety volume...", flush=True)
    result = fn.remote()
    print(json.dumps(result, indent=2))
    print("PUSH DONE")

if __name__ == "__main__":
    main()
```

Run: `.venv-sae/bin/python scripts/saemap_push_model.py`
Expected: a JSON record with `downloaded: true` (or `already_present: true` on re-run) and `path: /data/models/saemap_9b`, then `PUSH DONE`. This is the only "heavy" infra step here and just snapshots weights to the volume; it does not run the GPU.

- [ ] **Step 4: Write `game_theory_llm/saemap/remote.py`** (the local client)

```python
"""Local client over the Modal SaemapWorker (safety app, A100-80GB).

Replaces the original local MPS model.py: the heavy 9B forward passes run
remotely; this module marshals prompts out and numpy back. The SAE math stays
local (callers encode the returned residuals with QwenScopeSAE themselves).
"""
from __future__ import annotations
import numpy as np
import modal

APP = "safety"
CLS = "SaemapWorker"
MODEL_NAME = "saemap_9b"

_handle = None

def _worker():
    global _handle
    if _handle is None:
        Cls = modal.Cls.from_name(APP, CLS)          # mirrors run_full_pipeline.py:125
        _handle = Cls(model_name=MODEL_NAME)
    return _handle

def extract_residuals(prompts, layers, completion=None) -> np.ndarray:
    """[N, len(layers), D_MODEL] float32 residuals (mean-pooled over completion
    span if given, else over all prompt tokens)."""
    out = _worker().extract_residuals.remote(list(prompts), list(layers), completion)
    return np.asarray(out, dtype=np.float32)

def pcoop(prompts, coop_letters, fewshot="") -> np.ndarray:
    """[N] normalized P(coop) over {A,B}."""
    out = _worker().pcoop.remote(list(prompts), list(coop_letters), fewshot)
    return np.asarray(out, dtype=np.float32)

def causal_pcoop(prompts, coop_letters, layer, vec, fewshot="") -> np.ndarray:
    """[N] P(coop) with `vec` (4096-d) added at self.layers[layer]. `vec` may be a
    numpy array, torch tensor, or list — coerced to a plain float list for transport."""
    vec_list = np.asarray(vec, dtype=np.float32).reshape(-1).tolist()
    out = _worker().causal_pcoop.remote(
        list(prompts), list(coop_letters), int(layer), vec_list, fewshot)
    return np.asarray(out, dtype=np.float32)
```

- [ ] **Step 5: Smoke the worker boot + the three methods (small inputs)**

These are the integration tests for the worker/client (no mocked unit test — the methods are GPU-bound). Run each and confirm the expected console output.

(a) Residual shape + completion contrast:
```bash
.venv-sae/bin/python - <<'PY'
from game_theory_llm.saemap import remote
import numpy as np
r = remote.extract_residuals(["The two firms must decide whether to cooperate."],
                             layers=[16])
print("resid shape:", r.shape)                       # (1, 1, 4096)
assert r.shape == (1, 1, 4096), r.shape
rc = remote.extract_residuals(["They met to decide."], [16], " I will cooperate.")
rd = remote.extract_residuals(["They met to decide."], [16], " I will defect.")
print("coop vs defect L2 dist:", float(np.linalg.norm(rc - rd)))
assert np.linalg.norm(rc - rd) > 0
print("EXTRACT SMOKE OK")
PY
```
Expected: `resid shape: (1, 1, 4096)`, a positive coop-vs-defect distance, `EXTRACT SMOKE OK`.

(b) P(coop) is a probability:
```bash
.venv-sae/bin/python - <<'PY'
from game_theory_llm.saemap import remote
p = remote.pcoop(["Decide now. either: <decision>A</decision> or <decision>B</decision>."], ["A"])
print("p_coop:", round(float(p[0]), 3))
assert 0.0 <= p[0] <= 1.0
print("PCOOP SMOKE OK")
PY
```
Expected: a `p_coop:` value in [0,1], `PCOOP SMOKE OK`.

(c) Causal add moves the readout (and is removed afterward):
```bash
.venv-sae/bin/python - <<'PY'
import numpy as np
from game_theory_llm.saemap import remote
from game_theory_llm.saemap.sae import QwenScopeSAE
prompt = "Decide now. either: <decision>A</decision> or <decision>B</decision>."
base = remote.pcoop([prompt], ["A"])[0]
sae = QwenScopeSAE.load(16)
vec = 8.0 * sae.decoder_col(0, unit=True).numpy()    # any feature, large alpha
moved = remote.causal_pcoop([prompt], ["A"], layer=16, vec=vec)[0]
after = remote.pcoop([prompt], ["A"])[0]             # hook must be gone
print("base/moved/after:", round(float(base),3), round(float(moved),3), round(float(after),3))
assert abs(after - base) < 1e-4, "hook not removed"
print("CAUSAL SMOKE OK")
PY
```
Expected: `base/moved/after:` three numbers; `after ≈ base` (hook removed); `CAUSAL SMOKE OK`. (`moved` may or may not differ for an arbitrary feature — the point here is plumbing, not effect size.)

- [ ] **Step 6: Reconstruction-fidelity check (REMOTE residuals + LOCAL SAE; validates layer path + orientation)**

```bash
.venv-sae/bin/python - <<'PY'
import torch
from game_theory_llm.saemap import remote
from game_theory_llm.saemap.sae import QwenScopeSAE
texts = [
    "A long passage about two rival companies negotiating a risky deal.",
    "Harmony and cooperation benefited both villages for years.",
]
r = remote.extract_residuals(texts, layers=[16])      # [2, 1, 4096]
resid = torch.tensor(r[:, 0, :], dtype=torch.float32)  # [2, 4096]
sae = QwenScopeSAE.load(16)
ve = sae.variance_explained(resid)
print("variance_explained@L16:", round(ve, 3))
assert ve > 0.5, f"low recon fidelity ({ve}) -> wrong layer path/orientation"
print("RECON OK")
PY
```
Expected: `variance_explained@L16:` > 0.5, `RECON OK`. If `ve < 0.5`, the worker's `self.layers[L]` block output does not match the SAE's "layer L" residual locus: in the worker, try capturing the block *input* (`inp[0]`) instead of output, or verify `_discover_layers` returned `model.model.layers` (not a wrapper); re-run until fidelity is high and record the correct capture point in `_residuals_one`. (Use more than 2 texts if `den` is tiny.)

- [ ] **Step 7: Commit**

```bash
git add game_theory_llm/steering/modal_app.py scripts/saemap_push_model.py game_theory_llm/saemap/remote.py
git commit -m "saemap(modal): SaemapWorker + saemap_image + push 9B to volume + local remote client + recon fidelity"
```

---

## Task 4: Corpus builders (`corpus.py`) — LOCAL, device-agnostic

> Copied verbatim from the local plan; unchanged in Modal v2 (no model dependency).

**Files:**
- Create: `game_theory_llm/saemap/corpus.py`
- Test: `tests/saemap/test_corpus.py`

**Interfaces:**
- Consumes: `paths.PD_TRAIN, PD_EVAL, PD_SWAP, STORIES_DIR, NONGAME_FILES, DILEMMA_GAMES, NONDILEMMA_GAMES, ALL_GAMES`.
- Produces:
  - `COOP_CUE = " Decision made: I will cooperate and honor the agreement."`
  - `DEFECT_CUE = " Decision made: I will defect and break the agreement."`
  - `load_jsonl(path) -> list[dict]`
  - `decision_pairs(path=PD_TRAIN, limit=None) -> list[dict]` each `{"id","prompt","coop_letter","coop_cue","defect_cue"}` (coop_cue/defect_cue are the fixed templates; included per-row for convenience).
  - `pd_eval_set(limit=None) -> list[dict]` each `{"id","prompt","coop_letter"}` (from PD_EVAL; for P(coop)).
  - `game_texts(games: list[str], n_per_game=None) -> list[str]` (reads `content`).
  - `nongame_texts(n=None) -> list[str]` (reads `prompt`).
  - `recognition_sets(n_per_game=120) -> dict` → `{"dilemma":[str], "nondilemma":[str], "game":[str], "nongame":[str]}`.

- [ ] **Step 1: Write the failing test** (`tests/saemap/test_corpus.py`)

```python
from game_theory_llm.saemap import corpus, paths

def test_decision_pairs_have_letter_and_cues():
    rows = corpus.decision_pairs(limit=5)
    assert len(rows) == 5
    r = rows[0]
    assert r["coop_letter"] in {"A", "B"}
    assert "cooperate" in r["coop_cue"] and "defect" in r["defect_cue"]
    assert r["prompt"].strip().endswith(".")  # full prompt text present

def test_recognition_sets_partition_games():
    s = corpus.recognition_sets(n_per_game=10)
    assert len(s["dilemma"]) == 10 * len(paths.DILEMMA_GAMES)
    assert len(s["nondilemma"]) == 10 * len(paths.NONDILEMMA_GAMES)
    assert s["nongame"] and all(isinstance(t, str) for t in s["nongame"])
    # dilemma and nondilemma texts are disjoint
    assert not (set(s["dilemma"]) & set(s["nondilemma"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_corpus.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (no `corpus`).

- [ ] **Step 3: Write `game_theory_llm/saemap/corpus.py`**

```python
"""Corpus builders for the decision and recognition tracks."""
from __future__ import annotations
import json
from .paths import (PD_TRAIN, PD_EVAL, STORIES_DIR, NONGAME_FILES,
                    DILEMMA_GAMES, NONDILEMMA_GAMES)

COOP_CUE = " Decision made: I will cooperate and honor the agreement."
DEFECT_CUE = " Decision made: I will defect and break the agreement."

def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def decision_pairs(path=PD_TRAIN, limit=None):
    rows = load_jsonl(path)
    if limit:
        rows = rows[:limit]
    return [{"id": r["story_id"], "prompt": r["prompt"],
             "coop_letter": r["coop_choice"],
             "coop_cue": COOP_CUE, "defect_cue": DEFECT_CUE} for r in rows]

def pd_eval_set(limit=None):
    rows = load_jsonl(PD_EVAL)
    if limit:
        rows = rows[:limit]
    return [{"id": r["story_id"], "prompt": r["prompt"],
             "coop_letter": r["coop_choice"]} for r in rows]

def game_texts(games, n_per_game=None):
    out = []
    for g in games:
        texts = []
        for fp in sorted(STORIES_DIR.glob(f"{g}__*.jsonl")):
            for r in load_jsonl(fp):
                texts.append(r["content"])
        out.extend(texts[:n_per_game] if n_per_game else texts)
    return out

def nongame_texts(n=None):
    out = []
    for fp in NONGAME_FILES:
        if fp.exists():
            out.extend(r["prompt"] for r in load_jsonl(fp))
    return out[:n] if n else out

def recognition_sets(n_per_game=120):
    dil = game_texts(DILEMMA_GAMES, n_per_game)
    nondil = game_texts(NONDILEMMA_GAMES, n_per_game)
    game_all = game_texts(DILEMMA_GAMES + NONDILEMMA_GAMES +
                          ["battle_of_the_sexes", "matching_pennies"], n_per_game)
    nong = nongame_texts(len(game_all))
    return {"dilemma": dil, "nondilemma": nondil, "game": game_all, "nongame": nong}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_corpus.py -v`
Expected: 2 passed. (Needs the real data files from the path map; they exist.)

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/saemap/corpus.py tests/saemap/test_corpus.py
git commit -m "saemap(modal): decision cue-pair + recognition class-set corpus builders + tests"
```

---

## Task 5: Step-0 sanity gate (`saemap_sanity.py`) — HARD GATE (rewritten to use `remote`)

**Files:**
- Create: `scripts/saemap_sanity.py`

**Interfaces:**
- Consumes: `remote.pcoop`, `corpus.pd_eval_set`.
- Produces: `data/runs/saemap_9b/sanity.json` with `{"n", "mean_p_coop", "std_p_coop", "ab_mass_mean", "fewshot_used", "pass": bool}`. Prints `GATE PASS`/`GATE FAIL`.

Gate criteria (unchanged from spec §8): the readout must be usable before investing. (1) on average A/B together hold most of the next-token mass (`ab_mass_mean > 0.5`); (2) P(coop) is non-degenerate and responsive — `std_p_coop > 0.05` across scenarios (the decision varies with content, not pinned at 0/1/0.5). Try zero-shot first; if `ab_mass_mean ≤ 0.5`, retry with a 2-shot prefix and record `fewshot_used=True`.

**Modal-v2 change:** `ab_mass` (the A+B next-token mass) is now computed by a small worker helper so we don't ship logits across the boundary. Add this `@modal.method()` to `SaemapWorker` (alongside the others in Task 3 Step 2):

```python
    @modal.method()
    def ab_mass(self, prompts: list[str], fewshot: str = "") -> list:
        """Fraction of next-token mass on {A,B} after '\\n<decision>' (Step-0 gate)."""
        import torch
        out = []
        for p in prompts:
            text = fewshot + p + "\n<decision>"
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                probs = self.model(ids).logits[0, -1].softmax(-1)
            out.append((probs[self.tid_A] + probs[self.tid_B]).item())
        return out
```

and expose it in `remote.py`:

```python
def ab_mass(prompts, fewshot="") -> np.ndarray:
    out = _worker().ab_mass.remote(list(prompts), fewshot)
    return np.asarray(out, dtype=np.float32)
```

- [ ] **Step 1: Write `scripts/saemap_sanity.py`**

```python
"""Step-0 HARD GATE: is P(coop) a usable, responsive readout on 9B-Base?

Runs entirely against the Modal SaemapWorker via game_theory_llm.saemap.remote.
"""
import json, statistics
from game_theory_llm.saemap import remote, corpus, paths

FEWSHOT = (
    "You are deciding in a strategic scenario. Output your choice as a single letter.\n"
    "Example 1 ... <decision>A</decision>\n"
    "Example 2 ... <decision>B</decision>\n\n")

def run(fewshot):
    rows = corpus.pd_eval_set(limit=40)
    prompts = [r["prompt"] for r in rows]
    coop = [r["coop_letter"] for r in rows]
    ps = remote.pcoop(prompts, coop, fewshot).tolist()
    masses = remote.ab_mass(prompts, fewshot).tolist()
    return ps, masses

def main():
    paths.ensure_run_dirs()
    fewshot, used = "", False
    ps, masses = run(fewshot)
    if statistics.mean(masses) <= 0.5:
        fewshot, used = FEWSHOT, True
        ps, masses = run(fewshot)
    rec = {"n": len(ps), "mean_p_coop": statistics.mean(ps),
           "std_p_coop": statistics.pstdev(ps),
           "ab_mass_mean": statistics.mean(masses), "fewshot_used": used}
    rec["pass"] = rec["ab_mass_mean"] > 0.5 and rec["std_p_coop"] > 0.05
    (paths.RUN_DIR / "sanity.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print("GATE PASS" if rec["pass"] else "GATE FAIL")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `.venv-sae/bin/python scripts/saemap_sanity.py`
Expected: prints the record and `GATE PASS`. If `GATE FAIL` because `ab_mass_mean` is low even with few-shot, widen the candidate letter token-ids in the worker (handle `" A"`/`"A"` variants in `load()` where `tid_A`/`tid_B` are resolved) and re-run; if `std_p_coop` ≤ 0.05 (decision pinned regardless of scenario), STOP and report — the base model does not encode a content-responsive decision and the spec's Step-0 gate has failed; surface this to the user before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/saemap_sanity.py game_theory_llm/steering/modal_app.py game_theory_llm/saemap/remote.py
git commit -m "saemap(modal): Step-0 sanity gate via remote (P(coop) usable + responsive)"
```

---

## Task 6: Extraction (`saemap_extract.py`) — cache SAE features per layer (rewritten to use `remote`)

**Files:**
- Create: `scripts/saemap_extract.py`

**Interfaces:**
- Consumes: `remote.extract_residuals`, `QwenScopeSAE`, `corpus.decision_pairs`, `corpus.recognition_sets`.
- Produces: cached tensors under `data/runs/saemap_9b/activations/L{layer}/` (LOCAL):
  - `decision_coop.pt`, `decision_defect.pt` — `[N_dec, D_SAE]` (SAE features over the cue span).
  - `recog_{dilemma,nondilemma,game,nongame}.pt` — `[N, D_SAE]` (SAE features over scenario span).
  - `resid_decision_{coop,defect}.pt` — `[N_dec, D_MODEL]` raw residual means (for pass C decomposition).
  Plus `meta.json` with row ids/counts. Accepts `--layers 8,12,16,20,24 --limit N --n-per-game N`.

**Modal-v2 change vs original Task 6:** instead of `m.residuals(...)` / `m.residual_for_completion(...)` per text (MPS), we call `remote.extract_residuals(texts, layers, completion)` ONCE per (text-set, completion) and get back **all candidate layers at once** in a single remote round-trip, then SAE-encode each layer's slice locally with `QwenScopeSAE`. The raw residual means saved for pass C come straight from the same returned array (no second forward).

- [ ] **Step 1: Write `scripts/saemap_extract.py`**

```python
"""Cache SAE feature activations for decision (cue-span) + recognition (scenario-span).

Heavy 9B forward passes run remotely (SaemapWorker); SAE encoding is local.
extract_residuals returns [N, len(layers), D_MODEL] in ONE remote call, so we
fetch all candidate layers together and slice per layer locally.
"""
import argparse, json, torch
from game_theory_llm.saemap import remote, corpus, paths
from game_theory_llm.saemap.sae import QwenScopeSAE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-per-game", type=int, default=120)
    args = ap.parse_args()
    paths.ensure_run_dirs()
    layers = [int(x) for x in args.layers.split(",")]

    dec = corpus.decision_pairs(limit=args.limit)
    rec = corpus.recognition_sets(n_per_game=args.n_per_game)
    dec_prompts = [d["prompt"] for d in dec]

    # --- ONE remote call per text-set (all layers at once) ---
    # decision: same prompts, coop vs defect commitment cue, pool over cue span
    R_coop = torch.tensor(remote.extract_residuals(dec_prompts, layers, corpus.COOP_CUE))   # [Nd, L, D]
    R_def  = torch.tensor(remote.extract_residuals(dec_prompts, layers, corpus.DEFECT_CUE)) # [Nd, L, D]
    # recognition: scenario span, mean-pooled over all prompt tokens
    R_rec = {k: torch.tensor(remote.extract_residuals(rec[k], layers))                       # [Nk, L, D]
             for k in ["dilemma", "nondilemma", "game", "nongame"]}

    for li, L in enumerate(layers):
        outdir = paths.RUN_DIR / "activations" / f"L{L}"
        outdir.mkdir(parents=True, exist_ok=True)
        sae = QwenScopeSAE.load(L)
        rc = R_coop[:, li, :].float()            # [Nd, D_MODEL]
        rd = R_def[:, li, :].float()
        torch.save(sae.encode(rc), outdir / "decision_coop.pt")     # [Nd, D_SAE]
        torch.save(sae.encode(rd), outdir / "decision_defect.pt")
        torch.save(rc, outdir / "resid_decision_coop.pt")           # raw means for pass C
        torch.save(rd, outdir / "resid_decision_defect.pt")
        for key in ["dilemma", "nondilemma", "game", "nongame"]:
            r = R_rec[key][:, li, :].float()                        # [Nk, D_MODEL]
            torch.save(sae.encode(r), outdir / f"recog_{key}.pt")   # [Nk, D_SAE]
        print(f"L{L} cached: decision {len(dec_prompts)}, "
              f"recog {{ {', '.join(f'{k}:{len(rec[k])}' for k in rec)} }}")

    (paths.RUN_DIR / "activations" / "meta.json").write_text(json.dumps({
        "layers": layers, "n_decision": len(dec),
        "decision_ids": [d["id"] for d in dec],
        "n_recog": {k: len(rec[k]) for k in rec}}, indent=2))
    print("EXTRACT DONE")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Tiny smoke run (1 layer, limit 8)**

Run: `.venv-sae/bin/python scripts/saemap_extract.py --layers 16 --limit 8 --n-per-game 4`
Expected: prints `L16 cached: decision 8, recog { dilemma:12, nondilemma:8, ... }` then `EXTRACT DONE`. Verify shapes:
`.venv-sae/bin/python -c "import torch;a=torch.load('data/runs/saemap_9b/activations/L16/decision_coop.pt');print(a.shape)"` → `torch.Size([8, 65536])`.

- [ ] **Step 3: Full extraction (all candidate layers)**

Run: `.venv-sae/bin/python scripts/saemap_extract.py --layers 8,12,16,20,24 --limit 200 --n-per-game 120`
Expected: five `L{L} cached...` lines + `EXTRACT DONE`. (The forward passes run on the A100; the worker stays warm across the few `extract_residuals` calls — `scaledown_window=300`. If a single `extract_residuals` payload of all recog texts is large, the call still returns the small `[N, len(layers), 4096]` tensor; no volume write needed at these N.)

- [ ] **Step 4: Commit**

```bash
git add scripts/saemap_extract.py
git commit -m "saemap(modal): extract decision + recognition SAE features across candidate layers (remote residuals, local encode)"
```

---

## Task 7: Discovery (`discover.py` + `saemap_discover.py`) — LOCAL math

> `discover.py` (the pure-math unit) and its TDD test are copied verbatim from the local plan — they operate on cached numpy and are device-agnostic. `saemap_discover.py` is also unchanged: it reads the locally-cached `[N, D_SAE]`/`[N, D_MODEL]` tensors written by Task 6 and the local `QwenScopeSAE.W_dec`. No model calls happen in discovery. (The interpret pass added in Task 8 is the only part that touches the worker.)

**Files:**
- Create: `game_theory_llm/saemap/discover.py`
- Create: `scripts/saemap_discover.py`
- Test: `tests/saemap/test_discover.py`

**Interfaces:**
- Consumes: cached `[N, D_SAE]` tensors; `QwenScopeSAE` (for pass C `W_dec`).
- Produces (`discover.py`):
  - `diff_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray` — per-feature `mean(pos)-mean(neg)` `[D_SAE]`.
  - `l1_probe(X: np.ndarray, y: np.ndarray, C=0.05, seed=0) -> dict` → `{"coef":[D_SAE], "auc":float, "nonzero":[int], "n_nonzero":int}` (5-fold CV AUC on held-out).
  - `decompose_direction(direction: np.ndarray, W_dec: np.ndarray, topn=20) -> list[tuple[int,float]]` — features ranked by `|cosine(W_dec[:,f], direction)|`.
  - `top_features(scores: np.ndarray, k=10) -> list[int]`.
- Produces (`saemap_discover.py`): per contrast (`decision`, `recog_fine`, `recog_coarse`) and per layer, write `data/runs/saemap_9b/discover/{contrast}_L{layer}.json` with diff-of-means top-k, probe AUC + nonzero features, and (decision only) pass-C overlap. Pick the best layer per contrast by held-out AUC and write `discover/summary.json`.

- [ ] **Step 1: Write the failing test** (`tests/saemap/test_discover.py`)

```python
import numpy as np
from game_theory_llm.saemap import discover

def test_diff_of_means_finds_planted_feature():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 20)); neg = rng.normal(size=(50, 20))
    pos[:, 7] += 3.0                      # plant signal in feature 7
    s = discover.diff_of_means(pos, neg)
    assert int(np.argmax(s)) == 7

def test_l1_probe_separates_and_selects():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 30)); y = (rng.random(120) > 0.5).astype(int)
    X[y == 1, 3] += 2.5                   # feature 3 carries the label
    res = discover.l1_probe(X, y, C=0.2)
    assert res["auc"] > 0.8
    assert 3 in res["nonzero"]

def test_decompose_direction_ranks_aligned_column():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(8, 16)); direction = W[:, 5].copy()
    ranked = discover.decompose_direction(direction, W, topn=3)
    assert ranked[0][0] == 5 and ranked[0][1] > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_discover.py -v`
Expected: FAIL — no `discover` module.

- [ ] **Step 3: Write `game_theory_llm/saemap/discover.py`**

```python
"""Feature discovery: diff-of-means, L1 probe, steering-direction decomposition."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def diff_of_means(pos, neg):
    return pos.mean(0) - neg.mean(0)

def l1_probe(X, y, C=0.05, seed=0):
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())
    clf.fit(X, y)
    coef = clf.coef_[0]
    nz = np.nonzero(coef)[0]
    return {"coef": coef.tolist(), "auc": auc,
            "nonzero": nz.tolist(), "n_nonzero": int(nz.size)}

def decompose_direction(direction, W_dec, topn=20):
    d = direction / (np.linalg.norm(direction) + 1e-9)
    cols = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-9)
    cos = np.abs(d @ cols)                         # [D_SAE]
    idx = np.argsort(-cos)[:topn]
    return [(int(i), float(cos[i])) for i in idx]

def top_features(scores, k=10):
    return [int(i) for i in np.argsort(-np.abs(scores))[:k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_discover.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `scripts/saemap_discover.py`**

```python
"""Run discovery (A diff-of-means, B L1 probe, C decomposition) per contrast/layer."""
import json, numpy as np, torch
from game_theory_llm.saemap import discover, paths
from game_theory_llm.saemap.sae import QwenScopeSAE

def _load(layer, name):
    return torch.load(paths.RUN_DIR / "activations" / f"L{layer}" / f"{name}.pt").numpy()

def _contrast(pos, neg):
    X = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    dom = discover.diff_of_means(pos, neg)
    probe = discover.l1_probe(X, y, C=0.05)
    return {"auc": probe["auc"], "n_nonzero": probe["n_nonzero"],
            "diffmeans_top": discover.top_features(dom, 10),
            "probe_top": [int(i) for i in np.argsort(-np.abs(np.array(probe["coef"])))[:10]]}

def main():
    contrasts = {
        "decision": lambda L: (_load(L, "decision_coop"), _load(L, "decision_defect")),
        "recog_fine": lambda L: (_load(L, "recog_dilemma"), _load(L, "recog_nondilemma")),
        "recog_coarse": lambda L: (_load(L, "recog_game"), _load(L, "recog_nongame")),
    }
    summary = {}
    for name, getter in contrasts.items():
        best = None
        for L in paths.CANDIDATE_LAYERS:
            pos, neg = getter(L)
            res = _contrast(pos, neg)
            if name == "decision":                         # pass C: holistic direction
                rc = _load(L, "resid_decision_coop"); rd = _load(L, "resid_decision_defect")
                direction = rc.mean(0) - rd.mean(0)
                W_dec = QwenScopeSAE.load(L).W_dec.numpy()
                decomp = discover.decompose_direction(direction, W_dec, topn=10)
                res["decomp_top"] = decomp
                res["decomp_overlap"] = len(set(i for i, _ in decomp) & set(res["probe_top"]))
            res["layer"] = L
            (paths.RUN_DIR / "discover" / f"{name}_L{L}.json").write_text(json.dumps(res, indent=2))
            if best is None or res["auc"] > best["auc"]:
                best = res
        summary[name] = {"best_layer": best["layer"], "best_auc": best["auc"],
                         "n_nonzero": best["n_nonzero"]}
        print(f"{name}: best L{best['layer']} auc={best['auc']:.3f}")
    (paths.RUN_DIR / "discover" / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DISCOVER DONE")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run discovery**

Run: `.venv-sae/bin/python scripts/saemap_discover.py`
Expected: three `{contrast}: best L{n} auc=...` lines + `DISCOVER DONE`. Inspect `discover/summary.json`: decision and recog_fine AUC should be clearly > 0.5 (target ≥ ~0.8 for a clean signal). If recog_fine AUC ≈ 0.5, note that the dilemma/non-dilemma split is not linearly separable at these layers (report, don't fudge).

- [ ] **Step 7: Commit**

```bash
git add game_theory_llm/saemap/discover.py scripts/saemap_discover.py tests/saemap/test_discover.py
git commit -m "saemap(modal): discovery (diff-of-means, L1 probe, steering-direction decomposition)"
```

---

## Task 8: Interpretation (`interpret.py`) — name the features (rewritten to use `remote`)

**Files:**
- Create: `game_theory_llm/saemap/interpret.py`
- Modify: `scripts/saemap_discover.py` (append an interpret pass writing `interpret/top_features.json`)

**Interfaces:**
- Consumes: `remote.extract_residuals`, `QwenScopeSAE`, discovery `*_L{best}.json`.
- Produces: `max_activating_examples(layer, feature, texts, sae, topn=8) -> list[dict]` each `{"score":float, "snippet":str}` (the text whose scenario-span feature activation is highest), and `interpret/top_features.json` mapping each shortlisted feature → its top activating snippets, per contrast.

**Modal-v2 change:** the original signature was `max_activating_examples(model, sae, layer, feature, texts, topn=8)` and called `model.residuals([t], layer)` per text. We now (a) get residuals for all `texts` in ONE `remote.extract_residuals(texts, [layer])` call, then (b) encode locally and read `feature`. The signature changes to `max_activating_examples(layer, feature, texts, sae, topn=8)` (no `model` arg — residuals come from `remote`). The caller in `saemap_discover.py` is updated to match.

- [ ] **Step 1: Write `game_theory_llm/saemap/interpret.py`**

```python
"""Name shortlisted features by their max-activating examples.

Residuals for all texts come back in ONE remote call; SAE-encode is local.
"""
from __future__ import annotations
import torch
from . import remote

def max_activating_examples(layer, feature, texts, sae, topn=8):
    R = torch.tensor(remote.extract_residuals(texts, [layer]))   # [N, 1, D_MODEL]
    acts = sae.encode(R[:, 0, :].float())                        # [N, D_SAE]
    col = acts[:, feature]                                       # [N]
    order = torch.argsort(col, descending=True)[:topn].tolist()
    return [{"score": float(col[i].item()), "snippet": texts[i][:300]} for i in order]
```

- [ ] **Step 2: Append interpret pass to `scripts/saemap_discover.py`**

Add at the end of `main()` (before `print("DISCOVER DONE")`):

```python
    # --- interpret: label the shortlisted features of each contrast at its best layer ---
    from game_theory_llm.saemap.interpret import max_activating_examples
    from game_theory_llm.saemap import corpus
    pool = (corpus.game_texts(paths.DILEMMA_GAMES + paths.NONDILEMMA_GAMES, 40)
            + corpus.nongame_texts(120))
    labels = {}
    for name in ["decision", "recog_fine", "recog_coarse"]:
        L = summary[name]["best_layer"]
        res = json.loads((paths.RUN_DIR / "discover" / f"{name}_L{L}.json").read_text())
        sae = QwenScopeSAE.load(L)
        labels[name] = {str(f): max_activating_examples(L, f, pool, sae, 8)
                        for f in res["probe_top"][:5]}
    (paths.RUN_DIR / "interpret" / "top_features.json").write_text(json.dumps(labels, indent=2))
```

- [ ] **Step 3: Re-run discovery (now with interpret) and review**

Run: `.venv-sae/bin/python scripts/saemap_discover.py`
Expected: `DISCOVER DONE`; `interpret/top_features.json` exists. Read it and confirm the decision features' top snippets look cooperation/defection-related and the recog_fine features' snippets look dilemma-related (human read; per project rule do not regex-judge — eyeball or use an LLM judge if labeling formally).

- [ ] **Step 4: Commit**

```bash
git add game_theory_llm/saemap/interpret.py scripts/saemap_discover.py
git commit -m "saemap(modal): feature interpretation via max-activating examples (remote residuals, local encode)"
```

---

## Task 9: Causal test + mediation (`causal.py` + `saemap_causal.py`) — rewritten to use `remote`

**Files:**
- Create: `game_theory_llm/saemap/causal.py`
- Create: `scripts/saemap_causal.py`
- Test: `tests/saemap/test_causal_hook.py`

**Interfaces:**
- Consumes: `remote.causal_pcoop`, `remote.pcoop`, `QwenScopeSAE`, discovery shortlists, `corpus.pd_eval_set`.
- Produces (`causal.py`):
  - `decision_sweep(sae, layer, features, alphas, eval_rows, fewshot="") -> dict` — for each feature and α, mean P(coop) over eval_rows with `vec = α·unit(W_dec[:,f])` sent to `remote.causal_pcoop`.
  - `mediation(sae, layer, recog_features, eval_rows, alpha, fewshot="") -> dict` — ablate (subtract) the summed-unit recognition direction(s); report mean P(coop) shift vs the `remote.pcoop` baseline.
  - `random_feature_control(sae, layer, alphas, eval_rows, n=5, seed=0, fewshot="") -> dict`.
- Produces (`saemap_causal.py`): `data/runs/saemap_9b/causal/{decision_sweep,mediation,controls,verdict}.json` + a printed verdict.

**Modal-v2 change:** the original `causal.py` defined the forward hook locally (`add_vec_hook`) and a `p_coop_steered` that ran on MPS. In v2 the hook lives **in the worker** (`SaemapWorker.causal_pcoop`, Task 3) — `causal.py` becomes pure controller logic: it computes the decoder-column vectors locally (we have the SAE) and dispatches batches of prompts with one vector per call to `remote.causal_pcoop`. The TDD hook test is replaced by a `causal.py` arithmetic test against a fake `remote` (no GPU): it checks the controller builds the right vectors (`α·unit`, negated for ablation), batches eval rows correctly, and computes the means/shift. The worker's actual residual-add is integration-smoked in Task 3 Step 5(c).

- [ ] **Step 1: Write the failing controller test** (`tests/saemap/test_causal_hook.py`)

This test injects a fake `remote` so `causal.py` runs with no Modal/GPU. It asserts the controller (a) sends `α·unit(W_dec[:,f])` vectors of the right norm, (b) negates the direction for ablation, and (c) aggregates means/shift correctly.

```python
import numpy as np
import torch
from game_theory_llm.saemap import causal
from game_theory_llm.saemap.sae import QwenScopeSAE

class _FakeRemote:
    """Records calls; returns deterministic P(coop) = sigmoid(mean(vec))."""
    def __init__(self):
        self.calls = []
    def pcoop(self, prompts, coop_letters, fewshot=""):
        return np.full(len(prompts), 0.5, dtype=np.float32)
    def causal_pcoop(self, prompts, coop_letters, layer, vec, fewshot=""):
        self.calls.append({"layer": layer, "vec": np.asarray(vec, dtype=np.float32)})
        v = float(np.mean(vec))
        return np.full(len(prompts), 1.0 / (1.0 + np.exp(-v)), dtype=np.float32)

def _toy_sae(d_model=8, d_sae=32, k=4):
    sae = QwenScopeSAE.__new__(QwenScopeSAE)
    torch.manual_seed(0)
    sae.W_enc = torch.randn(d_sae, d_model); sae.W_dec = torch.randn(d_model, d_sae)
    sae.b_enc = torch.zeros(d_sae); sae.b_dec = torch.zeros(d_model)
    sae.k = k; sae.layer = 0
    return sae

def test_decision_sweep_sends_scaled_unit_vectors(monkeypatch):
    fake = _FakeRemote()
    monkeypatch.setattr(causal, "remote", fake)
    sae = _toy_sae()
    rows = [{"prompt": "p", "coop_letter": "A"}]
    out = causal.decision_sweep(sae, layer=0, features=[3], alphas=[2.0], eval_rows=rows)
    # one call; vec norm == |alpha| (unit direction scaled by alpha)
    assert len(fake.calls) == 1
    assert abs(np.linalg.norm(fake.calls[0]["vec"]) - 2.0) < 1e-4
    assert "3" in out and "2.0" in out["3"]

def test_mediation_negates_direction_and_reports_shift(monkeypatch):
    fake = _FakeRemote()
    monkeypatch.setattr(causal, "remote", fake)
    sae = _toy_sae()
    rows = [{"prompt": "p", "coop_letter": "A"}]
    med = causal.mediation(sae, layer=0, recog_features=[1], eval_rows=rows, alpha=4.0)
    assert med["baseline_mean"] == 0.5
    assert "shift" in med and "ablated_mean" in med
    # ablation vector points opposite the summed unit direction
    unit = sae.decoder_col(1, unit=True).numpy()
    sent = fake.calls[-1]["vec"]
    assert float(np.dot(sent, unit)) < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_causal_hook.py -v`
Expected: FAIL — no `causal` module.

- [ ] **Step 3: Write `game_theory_llm/saemap/causal.py`**

```python
"""Causal interventions (controller side): build feature-direction vectors locally
and dispatch them to the Modal worker's residual-add P(coop) readout.

The residual-ADD hook lives in SaemapWorker.causal_pcoop (steering's
make_steering_hook); here we only compute vectors and aggregate results.
"""
from __future__ import annotations
import numpy as np
import torch
from . import remote

def _prompts(eval_rows):
    return ([r["prompt"] for r in eval_rows],
            [r["coop_letter"] for r in eval_rows])

def decision_sweep(sae, layer, features, alphas, eval_rows, fewshot=""):
    prompts, coop = _prompts(eval_rows)
    res = {}
    for f in features:
        unit = sae.decoder_col(f, unit=True).numpy()        # [D_MODEL], ||=1
        res[str(f)] = {}
        for a in alphas:
            vec = (float(a) * unit)
            ps = remote.causal_pcoop(prompts, coop, layer, vec, fewshot)
            res[str(f)][str(a)] = float(np.mean(ps))
    return res

def mediation(sae, layer, recog_features, eval_rows, alpha, fewshot=""):
    prompts, coop = _prompts(eval_rows)
    base = remote.pcoop(prompts, coop, fewshot)
    direction = torch.stack([sae.decoder_col(f, unit=True) for f in recog_features]).sum(0)
    direction = (direction / direction.norm()).numpy()      # unit summed direction
    vec = (-float(alpha) * direction)                       # ABLATE: subtract recognition
    abl = remote.causal_pcoop(prompts, coop, layer, vec, fewshot)
    base_m, abl_m = float(np.mean(base)), float(np.mean(abl))
    return {"baseline_mean": base_m, "ablated_mean": abl_m,
            "shift": abl_m - base_m, "alpha": alpha,
            "recog_features": list(recog_features)}

def random_feature_control(sae, layer, alphas, eval_rows, n=5, seed=0, fewshot=""):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randint(0, sae.W_dec.shape[1], (n,), generator=g).tolist()
    return decision_sweep(sae, layer, feats, alphas, eval_rows, fewshot)
```

- [ ] **Step 4: Run the controller test to verify it passes**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_causal_hook.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write `scripts/saemap_causal.py`**

```python
"""Decision dose-response sweep + recognition->decision mediation + controls.

All P(coop) reads run on the Modal worker (remote.causal_pcoop / remote.pcoop);
vectors are computed locally from the SAE decoder columns.
"""
import json, numpy as np
from scipy import stats
from game_theory_llm.saemap.sae import QwenScopeSAE
from game_theory_llm.saemap import causal, corpus, paths

ALPHAS = [-8, -4, 0, 4, 8]

def monotonic_trend(alphas, means):
    return float(stats.spearmanr(alphas, means).correlation)

def main():
    summary = json.loads((paths.RUN_DIR / "discover" / "summary.json").read_text())
    sanity = json.loads((paths.RUN_DIR / "sanity.json").read_text())
    # mirror the Step-0 gate's few-shot choice so the readout matches discovery
    fewshot = ""
    if sanity.get("fewshot_used"):
        from scripts.saemap_sanity import FEWSHOT
        fewshot = FEWSHOT

    eval_rows = corpus.pd_eval_set(limit=40)

    # decision sweep at decision-track best layer
    Ld = summary["decision"]["best_layer"]
    saed = QwenScopeSAE.load(Ld)
    dec = json.loads((paths.RUN_DIR / "discover" / f"decision_L{Ld}.json").read_text())
    top_feats = dec["probe_top"][:3]
    sweep = causal.decision_sweep(saed, Ld, top_feats, ALPHAS, eval_rows, fewshot)
    ctrl = causal.random_feature_control(saed, Ld, ALPHAS, eval_rows, n=5, fewshot=fewshot)
    (paths.RUN_DIR / "causal" / "decision_sweep.json").write_text(json.dumps(
        {"layer": Ld, "features": top_feats, "alphas": ALPHAS, "sweep": sweep}, indent=2))
    (paths.RUN_DIR / "causal" / "controls.json").write_text(json.dumps(
        {"layer": Ld, "alphas": ALPHAS, "random_features": ctrl}, indent=2))

    # mediation at recognition-fine best layer
    Lr = summary["recog_fine"]["best_layer"]
    saer = QwenScopeSAE.load(Lr)
    rec = json.loads((paths.RUN_DIR / "discover" / f"recog_fine_L{Lr}.json").read_text())
    med = causal.mediation(saer, Lr, rec["probe_top"][:3], eval_rows, alpha=8, fewshot=fewshot)
    (paths.RUN_DIR / "causal" / "mediation.json").write_text(json.dumps(med, indent=2))

    # verdict
    rhos = {f: monotonic_trend(ALPHAS, [sweep[str(f)][str(a)] for a in ALPHAS]) for f in top_feats}
    ctrl_rhos = [monotonic_trend(ALPHAS, [ctrl[f][str(a)] for a in ALPHAS]) for f in ctrl]
    verdict = {
        "decision_feature_spearman": rhos,
        "max_abs_decision_rho": max(abs(v) for v in rhos.values()),
        "max_abs_control_rho": max(abs(v) for v in ctrl_rhos),
        "mediation_shift": med["shift"],
        "decision_causal_pass": max(abs(v) for v in rhos.values()) > 0.8
                                 and max(abs(v) for v in ctrl_rhos) < 0.5,
        "mediation_pass": abs(med["shift"]) > 0.05,
    }
    (paths.RUN_DIR / "causal" / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    print("CAUSAL DONE")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the causal test**

Run: `.venv-sae/bin/python scripts/saemap_causal.py`
Expected: prints a verdict JSON + `CAUSAL DONE`. A positive result: `decision_causal_pass=true` (top decision feature moves P(coop) monotonically, |ρ|>0.8, while random controls stay flat, |ρ|<0.5) and `mediation_pass=true` (ablating recognition features shifts P(coop) by >0.05). Report whatever comes out honestly — a flat sweep or null mediation is a real (publishable) negative, not a bug to mask.

- [ ] **Step 7: Commit**

```bash
git add game_theory_llm/saemap/causal.py scripts/saemap_causal.py tests/saemap/test_causal_hook.py
git commit -m "saemap(modal): causal decision sweep + recognition mediation + random-feature controls (remote readout)"
```

---

## Task 10: Results writeup + research-log entry

> Copied verbatim from the local plan (consumes only the local JSON artifacts; device-agnostic).

**Files:**
- Create: `docs/results/saemap_9b_result.md`
- Use the `/log` skill to append a research-notebook entry.

**Interfaces:**
- Consumes: `discover/summary.json`, `interpret/top_features.json`, `causal/verdict.json`, `sanity.json`.

- [ ] **Step 1: Write `docs/results/saemap_9b_result.md`**

Include: the Step-0 gate outcome (was few-shot needed; mean/std P(coop)); discovery AUCs per contrast and best layers; the named top features (decision + recog_fine) with one representative max-activating snippet each; the decision dose-response (P(coop) at each α for the top feature) vs the random-feature control; the mediation shift; and the verdict booleans. State the honest conclusion (mechanistic confirmation, partial, or null) and whether to trigger the staged 27B replication. Copy the doc to the main mirror: `cp docs/results/saemap_9b_result.md /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/docs/results/saemap_9b_result.md`.

- [ ] **Step 2: Append a `/log` research-notebook entry** summarizing the hypothesis, method (now Modal-hosted 9B), and result.

- [ ] **Step 3: Commit**

```bash
git add docs/results/saemap_9b_result.md
git commit -m "saemap(modal): 9B results writeup (discovery + causal + mediation)"
```

---

## Staged follow-up (NOT in this plan; gated on a positive 9B result)

Replication on `Qwen3.5-27B-Base` + `SAE-Res-Qwen3.5-27B-W80K-L0_100` is now a near-trivial extension of the Modal path: push the 27B weights to the `safety` volume under `/data/models/saemap_27b` (`download_saemap_model` with a different `hf_id`/`local_name`), add a 27B `SaemapWorker` variant (or pass `model_name="saemap_27b"`), and adjust `paths.py` ids, `D_MODEL=5120`, `D_SAE=81920`, `TOPK=100`, and the candidate-layer list (27B has 64 layers → sweep ~{16,24,32,40,48}). The worker is already on `A100-80GB`, so no infra change is needed. Only spin this up if `causal/verdict.json` shows `decision_causal_pass` (and ideally `mediation_pass`). This is the standing ≥2-model validation bar.

---

## Self-Review

**Spec coverage (recognition + decision tracks, mediation, controls, Step-0 gate — all preserved):**
- §2 decision track → Tasks 4,6,7,9. ✓ (cue-pair residuals via `remote.extract_residuals(..., completion=cue)`; SAE-encode local; sweep via `remote.causal_pcoop`.)
- §2 recognition track (fine + coarse) → Tasks 4,6,7. ✓ (scenario-span residuals via `remote.extract_residuals(texts, layers)`.)
- §3 env → Task 1 (LOCAL `.venv-sae` + `modal` client) + Global Constraints (Modal `safety` app, A100-80GB, `saemap_image` transformers>=5.10, model on volume). ✓
- §4 module layout → all modules present; `model.py` correctly REPLACED by `remote.py` + `SaemapWorker`. ✓
- §5a cue-pair contrast + label-swap → Task 4 (`decision_pairs`, cues); label-swap available via `PD_SWAP` (optional control reported in Task 10). ✓
- §5b recognition contrasts → Task 4 `recognition_sets`. ✓
- §5c layer sweep → Tasks 6 (extracts all candidate layers in one call), 7. ✓
- §6 P(coop) readout + intervention + mediation + controls (label-swap, random-feature, recon fidelity) → Task 3 (`pcoop`/`causal_pcoop` + recon@Step6), Task 9 (sweep/mediation/random-feature). ✓
- §7 success criteria → Task 9 verdict + Task 7 AUCs. ✓
- §8 Step-0 gate (HARD) → Task 5 (`remote.pcoop` + worker `ab_mass`); unchanged criteria (`ab_mass_mean>0.5`, `std_p_coop>0.05`). ✓

**Placeholder scan:** no TBD/TODO; every code step is complete and runnable. The worker methods, the client, and every caller use real, consistent signatures. The recon-fidelity run is concretely specified (remote residuals → local `variance_explained > 0.5`). ✓

**Type/signature consistency (worker ↔ client ↔ callers):**
- Worker `extract_residuals(prompts, layers, completion=None) -> [N, len(layers), 4096]` ↔ client `remote.extract_residuals(prompts, layers, completion=None) -> np.ndarray [N, len(layers), 4096]` ↔ callers: Task 6 (`completion=COOP_CUE/DEFECT_CUE` and no completion for recog), Task 8 (`[layer]`), Task 3 Step 6 (recon). ✓
- Worker `pcoop(prompts, coop_letters, fewshot="") -> [N]` ↔ client `remote.pcoop(...)` ↔ callers: Task 5 (sanity), Task 9 (`mediation` baseline). ✓
- Worker `causal_pcoop(prompts, coop_letters, layer, vec, fewshot="") -> [N]` ↔ client `remote.causal_pcoop(...)` (coerces vec→list) ↔ callers: Task 9 `decision_sweep` (`α·unit`), `mediation` (`-α·unit`), `random_feature_control`. ✓
- Worker `ab_mass(prompts, fewshot="") -> [N]` ↔ client `remote.ab_mass(...)` ↔ caller Task 5. ✓
- SAE-encoded tensors are `[N, D_SAE]` and raw residual means `[N, D_MODEL]` everywhere downstream (Tasks 6→7→8→9), matching the local plan. `decoder_col(f, unit=True)` returns `[D_MODEL]`, transported as a flat float list to `causal_pcoop`. ✓
- `interpret.max_activating_examples` signature changed from `(model, sae, layer, feature, texts, topn)` → `(layer, feature, texts, sae, topn)`; the only caller (Task 8 Step 2 appended block) is updated to match. ✓

**Reuse correctness (steering helpers):**
- `make_steering_hook(direction, alpha, raw_norm)` (application.py:18) adds `(alpha*raw_norm)*direction` and handles tuple outputs — called with `alpha=1.0, raw_norm=1.0` so it is a plain `+vec` add (feature-clamp/ablate). Signature matches this architecture exactly; no shim needed. ✓
- `_worker_load_impl(self, model_name)` + `_discover_layers(model)` reused unchanged; `model_name="saemap_9b"` resolves to `/data/models/saemap_9b` on the volume via `model_local_path`. ✓
- `download_saemap_model` mirrors `download_model_once` (snapshot_download → `volume.commit()`), writing under a stable `local_name` so the worker path is independent of the HF repo id. ✓

**Concerns surfaced for the implementer:**
1. **`saemap_image` transformers pin** is the one real risk: `>=5.10` is required (Gemma's image at `>=4.46` cannot load Qwen3.5; config nests under `text_config`). If `>=5.10` resolves to a release that breaks `_worker_load_impl`, pin the exact local-validated `transformers==5.12.1`. The existing `image` is untouched, so Gemma workers are unaffected either way.
2. **Layer-path / capture-point validation is deferred to runtime** (Task 3 Step 6 recon-fidelity > 0.5). `_discover_layers` is expected to return `model.model.layers` (32 blocks) for Qwen3.5-9B-Base; if the SAE locus is the block *input* rather than output, switch `_residuals_one` to capture `inp[0]`. This is the single empirical unknown and is gated by an explicit assertion before any discovery runs.
3. **Token-id resolution** for A/B is done once at `@modal.enter()`; if `ab_mass_mean` is low the fix (widen to `" A"`/`"A"` variants) is localized to `load()` and called out in Task 5 Step 2.
