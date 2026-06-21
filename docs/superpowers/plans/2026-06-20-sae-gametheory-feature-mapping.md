# SAE Feature-Mapping of Game-Theoretic Reasoning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the cooperate↔defect direction and a game-theoretic "recognition" representation into interpretable Qwen-Scope SAE features on `Qwen3.5-9B-Base`, then causally test them (clamp → P(coop) dose-response; ablate recognition → mediation), entirely locally on an M4.

**Architecture:** A small `game_theory_llm/saemap/` package: a pure-math SAE loader (`sae.py`), a model wrapper that reads residuals + a P(coop) readout (`model.py`), corpus builders (`corpus.py`), discovery (`discover.py`), interpretation (`interpret.py`), and causal hooks (`causal.py`). Thin scripts orchestrate env setup, a hard Step-0 sanity gate, extraction, discovery, and the causal/mediation tests. Pure logic is TDD'd against synthetic tensors; model-dependent steps are scripts with explicit assertion checks on their output JSON.

**Tech Stack:** Python 3.11 venv (`.venv-sae`), `transformers` (Qwen3.5 support), `torch` (MPS), `huggingface_hub`, `safetensors`, `scikit-learn`, `numpy`, `pytest`.

## Global Constraints

- **Spec:** `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/docs/superpowers/specs/2026-06-18-sae-cooperation-feature-mapping-design.md`
- **Work in the worktree:** all code/commits on branch `feature/activation-steering` at `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/` (cwd for every command below).
- **Interpreter:** use `.venv-sae/bin/python` for all model/SAE code (Python 3.11). Do NOT use system `python3` (3.9, cannot load Qwen3.5). Pure-math unit tests also run under `.venv-sae/bin/python -m pytest`.
- **Model:** `Qwen/Qwen3.5-9B-Base`. **SAE repo:** `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` (per-layer `.pt`: keys `W_enc` (65536,4096), `W_dec` (4096,65536), `b_enc` (65536,), `b_dec` (4096,); d_model 4096, d_sae 65536, **TopK=50 applied at runtime, no activation fn**).
- **SAE math:** `pre = resid @ W_enc.T + b_enc`; keep top-50 per token (zero the rest); `recon = acts @ W_dec.T + b_dec`; feature `f` decoder direction = `W_dec[:, f]`.
- **Layer indexing:** SAE "layer L" = output of transformer block L = `output_hidden_states` index `L+1`; forward hooks attach to `model.model.layers[L]`. Validated empirically by the reconstruction-fidelity check (Task 2).
- **Candidate layers:** `[8, 12, 16, 20, 24]`. **Memory rule:** never hold more than one SAE layer in memory at once (load per-layer); model bf16 ≈ 18 GB, one SAE ≈ 1.1 GB, peak < 34 GB.
- **Decision readout:** PD prompts already end with `...your decision, either: <decision>A</decision> or <decision>B</decision>...`. P(coop) = next-token prob of the cooperative letter after appending `<decision>`, normalized over {A,B}; `coop_choice` gives the cooperative letter; optional few-shot prefix toggled by the Step-0 gate.
- **No regex for quality/semantic judgments** (project rule); feature *labels* come from max-activating examples, optionally an LLM judge — never regex pattern-matching to claim a feature's meaning.
- **Run dir:** `data/runs/saemap_9b/` (under worktree).

---

## File Structure

```
game_theory_llm/saemap/
    __init__.py        # exports
    paths.py           # absolute paths to corpora (main + worktree), ids, layer list, run dir
    sae.py             # QwenScopeSAE: load/encode(topk)/decoder_col/reconstruct/variance_explained
    model.py           # QwenModel: load 9B (MPS); residuals(); residual_for_completion(); p_coop()
    corpus.py          # decision cue-pairs (+swap); recognition class sets (dilemma/nondilemma/game/nongame)
    discover.py        # diff_of_means; l1_probe; decompose_direction; held_out_auc
    interpret.py       # max_activating_examples
    causal.py          # add_vec_hook; ablate_feature_hook; decision_sweep; mediation; random_feature_control
scripts/
    saemap_setup_env.sh  # create .venv-sae + install deps + download model & candidate SAE layers
    saemap_sanity.py     # Step-0 HARD GATE: P(coop) non-degenerate + responsive
    saemap_extract.py    # cache decision cue-span + recognition scenario-span SAE features per layer
    saemap_discover.py   # run A/B/C per contrast -> discover/*.json + interpret top features
    saemap_causal.py     # decision sweep + mediation + controls -> causal/*.json + verdict
tests/saemap/
    __init__.py
    test_sae.py          # synthetic: encode shape, L0==50, reconstruct round-trip
    test_corpus.py       # cue-pair + label-swap + class-set membership
    test_discover.py     # diff_of_means + l1_probe recover a planted signal
```

---

## Task 1: Environment + paths module + weight download

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
# Create the Python 3.11 venv for SAE work and download model + candidate SAE layers.
set -euo pipefail
cd "$(dirname "$0")/.."          # worktree root

if [ ! -d .venv-sae ]; then
  python3.11 -m venv .venv-sae
fi
.venv-sae/bin/python -m pip install -q --upgrade pip
.venv-sae/bin/python -m pip install -q \
  "torch>=2.4" "transformers>=4.57" "huggingface_hub>=0.36" safetensors \
  "scikit-learn>=1.4" numpy pandas pytest

# Verify Qwen3.5 architecture is loadable (config only — cheap, no weights)
.venv-sae/bin/python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B-Base")
assert cfg.hidden_size == 4096, cfg.hidden_size
print("OK Qwen3.5-9B-Base config loads; hidden_size", cfg.hidden_size,
      "num_layers", cfg.num_hidden_layers)
PY

# Download candidate SAE layer files into the run cache (per-layer, ~1GB each)
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

# Pre-fetch the model weights (≈18GB) so later steps don't stall
.venv-sae/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from game_theory_llm.saemap import paths
snapshot_download(repo_id=paths.MODEL_ID)
print("OK model snapshot cached")
PY
echo "ENV READY"
```

- [ ] **Step 4: Run env setup**

Run: `bash scripts/saemap_setup_env.sh`
Expected: ends with `OK Qwen3.5-9B-Base config loads...`, five `downloaded .../layer{L}.sae.pt`, `OK model snapshot cached`, `ENV READY`.
If `AutoConfig` fails on the architecture, bump `transformers` to the newest release and re-run (Qwen3.5 needs a recent version). If the SAE filename differs, list the repo: `.venv-sae/bin/python -c "from huggingface_hub import list_repo_files; print([f for f in list_repo_files('Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50') if 'layer' in f][:5])"` and adjust the `fn` pattern.

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/saemap/__init__.py game_theory_llm/saemap/paths.py scripts/saemap_setup_env.sh
git commit -m "saemap: env setup, paths module, weight download"
```

---

## Task 2: SAE loader (`sae.py`) — pure math + reconstruction fidelity

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

This requires `model.py` (Task 3). **Defer the run to Task 3 Step 6**, but add the check now as a script block in Task 3. For this task, just confirm the math units pass.

- [ ] **Step 6: Commit**

```bash
git add game_theory_llm/saemap/sae.py tests/saemap/__init__.py tests/saemap/test_sae.py
git commit -m "saemap: Qwen-Scope SAE loader (topk encode, reconstruct) + tests"
```

---

## Task 3: Model wrapper (`model.py`) — residuals, completion pooling, P(coop)

**Files:**
- Create: `game_theory_llm/saemap/model.py`
- Test: `scripts/_smoke_model.py` (temporary integration smoke; deleted after)

**Interfaces:**
- Consumes: `paths.MODEL_ID`; `QwenScopeSAE` (for the recon check).
- Produces: `class QwenModel`:
  - `__init__(self, device="mps", dtype=torch.bfloat16)`
  - `residuals(self, prompts: list[str], layer: int, pool: str = "mean") -> torch.Tensor` → `[len(prompts), D_MODEL]` (pooled over all prompt tokens; `pool` in {"mean","last"}); reads `hidden_states[layer+1]`.
  - `residual_for_completion(self, prompt: str, completion: str, layer: int) -> torch.Tensor` → `[D_MODEL]` pooled over the completion tokens only.
  - `p_coop(self, prompt: str, coop_letter: str, fewshot: str = "") -> float` → normalized P(coop letter) over {A,B} after appending `<decision>`.
  - `_hidden(self, input_ids) -> tuple` helper returning hidden_states tuple (used by hooks in Task 9).

- [ ] **Step 1: Write `game_theory_llm/saemap/model.py`**

```python
"""Qwen3.5-9B-Base wrapper: residual extraction + P(coop) readout (MPS)."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .paths import MODEL_ID

class QwenModel:
    def __init__(self, device="mps", dtype=torch.bfloat16):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype, output_hidden_states=True).to(device).eval()

    @torch.no_grad()
    def residuals(self, prompts, layer, pool="mean"):
        out = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt").input_ids.to(self.device)
            hs = self.model(ids).hidden_states[layer + 1][0]   # [T, D_MODEL]
            v = hs.mean(0) if pool == "mean" else hs[-1]
            out.append(v.float().cpu())
        return torch.stack(out)

    @torch.no_grad()
    def residual_for_completion(self, prompt, completion, layer):
        pid = self.tok(prompt, return_tensors="pt").input_ids
        full = self.tok(prompt + completion, return_tensors="pt").input_ids
        start = pid.shape[1]
        hs = self.model(full.to(self.device)).hidden_states[layer + 1][0]  # [T, D]
        return hs[start:].mean(0).float().cpu()                            # completion span

    @torch.no_grad()
    def p_coop(self, prompt, coop_letter, fewshot=""):
        text = fewshot + prompt + "\n<decision>"
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(ids).logits[0, -1]                 # next-token logits
        def tid(letter):
            # first sub-token id of the bare letter as it follows "<decision>"
            return self.tok(letter, add_special_tokens=False).input_ids[0]
        a = logits[tid("A")].item()
        b = logits[tid("B")].item()
        pa, pb = torch.softmax(torch.tensor([a, b]), 0).tolist()
        return pa if coop_letter == "A" else pb
```

- [ ] **Step 2: Write the integration smoke (`scripts/_smoke_model.py`)**

```python
import torch
from game_theory_llm.saemap.model import QwenModel
from game_theory_llm.saemap.sae import QwenScopeSAE
from game_theory_llm.saemap import paths

m = QwenModel()
# residual shape
r = m.residuals(["The two firms must decide whether to cooperate."], layer=16)
assert r.shape == (1, paths.D_MODEL), r.shape
# completion pooling differs for coop vs defect cue
rc = m.residual_for_completion("They met to decide.", " I will cooperate.", 16)
rd = m.residual_for_completion("They met to decide.", " I will defect.", 16)
assert rc.shape == (paths.D_MODEL,) and (rc - rd).norm() > 0
# reconstruction fidelity at layer 16 (validates orientation + layer index)
big = m.residuals(["A long passage about two rival companies negotiating a risky deal."] * 1, 16)
sae = QwenScopeSAE.load(16)
ve = sae.variance_explained(torch.cat([big, m.residuals(
    ["Harmony and cooperation benefited both villages for years."], 16)], 0))
print("variance_explained@L16:", round(ve, 3))
assert ve > 0.5, f"low recon fidelity ({ve}) -> wrong layer index/orientation"
# p_coop returns a probability
p = m.p_coop("Decide now. either: <decision>A</decision> or <decision>B</decision>.", "A")
print("p_coop:", round(p, 3))
assert 0.0 <= p <= 1.0
print("SMOKE OK")
```

- [ ] **Step 3: Run the smoke**

Run: `.venv-sae/bin/python scripts/_smoke_model.py`
Expected: prints `variance_explained@L16:` > 0.5, a `p_coop:` value in [0,1], then `SMOKE OK`.
If `variance_explained` < 0.5, the layer→hidden_states offset or W_enc/W_dec orientation is wrong: try `hidden_states[layer]` (offset 0) and/or transpose; re-run until fidelity is high. Record the correct offset in `residuals`/`residual_for_completion` and in Task 9 hooks.

- [ ] **Step 4: Delete the smoke and commit**

```bash
rm scripts/_smoke_model.py
git add game_theory_llm/saemap/model.py
git commit -m "saemap: Qwen 9B model wrapper (residuals, completion pooling, p_coop) + verified recon fidelity"
```

---

## Task 4: Corpus builders (`corpus.py`)

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
git commit -m "saemap: decision cue-pair + recognition class-set corpus builders + tests"
```

---

## Task 5: Step-0 sanity gate (`saemap_sanity.py`) — HARD GATE

**Files:**
- Create: `scripts/saemap_sanity.py`

**Interfaces:**
- Consumes: `QwenModel.p_coop`, `corpus.pd_eval_set`.
- Produces: `data/runs/saemap_9b/sanity.json` with `{"n", "mean_p_coop", "std_p_coop", "ab_mass_mean", "fewshot_used", "pass": bool}`. Prints `GATE PASS`/`GATE FAIL`.

Gate criteria (the readout must be usable before investing): (1) on average A/B together hold most of the next-token mass (`ab_mass_mean > 0.5`); (2) P(coop) is non-degenerate and responsive — `std_p_coop > 0.05` across scenarios (the decision varies with content, not pinned at 0/1/0.5). Try zero-shot first; if `ab_mass_mean ≤ 0.5`, retry with a 2-shot prefix and record `fewshot_used=True`.

- [ ] **Step 1: Write `scripts/saemap_sanity.py`**

```python
"""Step-0 HARD GATE: is P(coop) a usable, responsive readout on 9B-Base?"""
import json, statistics, torch
from game_theory_llm.saemap.model import QwenModel
from game_theory_llm.saemap import corpus, paths

FEWSHOT = (
    "You are deciding in a strategic scenario. Output your choice as a single letter.\n"
    "Example 1 ... <decision>A</decision>\n"
    "Example 2 ... <decision>B</decision>\n\n")

def ab_mass(m, prompt, fewshot):
    text = fewshot + prompt + "\n<decision>"
    ids = m.tok(text, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        probs = m.model(ids).logits[0, -1].softmax(-1)
    ta = m.tok("A", add_special_tokens=False).input_ids[0]
    tb = m.tok("B", add_special_tokens=False).input_ids[0]
    return (probs[ta] + probs[tb]).item()

def run(fewshot):
    m = QwenModel()
    rows = corpus.pd_eval_set(limit=40)
    ps = [m.p_coop(r["prompt"], r["coop_letter"], fewshot) for r in rows]
    masses = [ab_mass(m, r["prompt"], fewshot) for r in rows]
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
Expected: prints the record and `GATE PASS`. If `GATE FAIL` because `ab_mass_mean` is low even with few-shot, widen the candidate letter token-ids (handle `" A"`/`"A"` variants) in `p_coop`/`ab_mass`; if `std_p_coop` ≤ 0.05 (decision pinned regardless of scenario), STOP and report — the base model does not encode a content-responsive decision and the spec's Step-0 gate has failed; surface this to the user before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/saemap_sanity.py
git commit -m "saemap: Step-0 sanity gate (P(coop) usable + responsive)"
```

---

## Task 6: Extraction (`saemap_extract.py`) — cache SAE features per layer

**Files:**
- Create: `scripts/saemap_extract.py`

**Interfaces:**
- Consumes: `QwenModel`, `QwenScopeSAE`, `corpus.decision_pairs`, `corpus.recognition_sets`.
- Produces: cached tensors under `data/runs/saemap_9b/activations/L{layer}/`:
  - `decision_coop.pt`, `decision_defect.pt` — `[N_dec, D_SAE]` (SAE features over the cue span).
  - `recog_{dilemma,nondilemma,game,nongame}.pt` — `[N, D_SAE]` (SAE features over scenario span).
  - `resid_decision_{coop,defect}.pt` — `[N_dec, D_MODEL]` raw residual means (for pass C decomposition).
  Plus `meta.json` with row ids/counts. Accepts `--layers 8,12,16,20,24 --limit N`.

- [ ] **Step 1: Write `scripts/saemap_extract.py`**

```python
"""Cache SAE feature activations for decision (cue-span) + recognition (scenario-span)."""
import argparse, json, torch
from game_theory_llm.saemap.model import QwenModel
from game_theory_llm.saemap.sae import QwenScopeSAE
from game_theory_llm.saemap import corpus, paths

def encode_texts(m, sae, texts, layer, completion=None):
    """Return [N, D_SAE] SAE features. If completion given, pool over completion span."""
    feats = []
    for t in texts:
        if completion is not None:
            r = m.residual_for_completion(t, completion, layer)      # [D_MODEL]
        else:
            r = m.residuals([t], layer)[0]                            # [D_MODEL]
        feats.append(sae.encode(r))
    return torch.stack(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-per-game", type=int, default=120)
    args = ap.parse_args()
    paths.ensure_run_dirs()
    layers = [int(x) for x in args.layers.split(",")]

    m = QwenModel()
    dec = corpus.decision_pairs(limit=args.limit)
    rec = corpus.recognition_sets(n_per_game=args.n_per_game)
    dec_prompts = [d["prompt"] for d in dec]

    for L in layers:
        outdir = paths.RUN_DIR / "activations" / f"L{L}"
        outdir.mkdir(parents=True, exist_ok=True)
        sae = QwenScopeSAE.load(L)
        # decision: same prompt, coop vs defect commitment cue, pool over cue span
        torch.save(encode_texts(m, sae, dec_prompts, L, corpus.COOP_CUE), outdir / "decision_coop.pt")
        torch.save(encode_texts(m, sae, dec_prompts, L, corpus.DEFECT_CUE), outdir / "decision_defect.pt")
        # raw residual means for pass C (holistic direction)
        rc = torch.stack([m.residual_for_completion(p, corpus.COOP_CUE, L) for p in dec_prompts])
        rd = torch.stack([m.residual_for_completion(p, corpus.DEFECT_CUE, L) for p in dec_prompts])
        torch.save(rc, outdir / "resid_decision_coop.pt")
        torch.save(rd, outdir / "resid_decision_defect.pt")
        # recognition: scenario span, mean-pooled
        for key in ["dilemma", "nondilemma", "game", "nongame"]:
            torch.save(encode_texts(m, sae, rec[key], L), outdir / f"recog_{key}.pt")
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
Expected: five `L{L} cached...` lines + `EXTRACT DONE`. (Minutes-scale on MPS; one SAE loaded at a time.)

- [ ] **Step 4: Commit**

```bash
git add scripts/saemap_extract.py
git commit -m "saemap: extract decision + recognition SAE features across candidate layers"
```

---

## Task 7: Discovery (`discover.py` + `saemap_discover.py`)

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
git commit -m "saemap: discovery (diff-of-means, L1 probe, steering-direction decomposition)"
```

---

## Task 8: Interpretation (`interpret.py`) — name the features

**Files:**
- Create: `game_theory_llm/saemap/interpret.py`
- Modify: `scripts/saemap_discover.py` (append an interpret pass writing `interpret/top_features.json`)

**Interfaces:**
- Consumes: `QwenModel`, `QwenScopeSAE`, discovery `*_L{best}.json`.
- Produces: `max_activating_examples(model, sae, layer, feature, texts, topn=8) -> list[dict]` each `{"score":float, "snippet":str}` (the text whose scenario-span feature activation is highest), and `interpret/top_features.json` mapping each shortlisted feature → its top activating snippets, per contrast.

- [ ] **Step 1: Write `game_theory_llm/saemap/interpret.py`**

```python
"""Name shortlisted features by their max-activating examples."""
from __future__ import annotations
import torch

def max_activating_examples(model, sae, layer, feature, texts, topn=8):
    scored = []
    for t in texts:
        r = model.residuals([t], layer)[0]        # [D_MODEL]
        act = sae.encode(r)[feature].item()
        scored.append((act, t))
    scored.sort(key=lambda x: -x[0])
    return [{"score": float(s), "snippet": t[:300]} for s, t in scored[:topn]]
```

- [ ] **Step 2: Append interpret pass to `scripts/saemap_discover.py`**

Add at the end of `main()` (before `print("DISCOVER DONE")`):

```python
    # --- interpret: label the shortlisted features of each contrast at its best layer ---
    from game_theory_llm.saemap.model import QwenModel
    from game_theory_llm.saemap.interpret import max_activating_examples
    from game_theory_llm.saemap import corpus
    m = QwenModel()
    pool = (corpus.game_texts(paths.DILEMMA_GAMES + paths.NONDILEMMA_GAMES, 40)
            + corpus.nongame_texts(120))
    labels = {}
    for name in ["decision", "recog_fine", "recog_coarse"]:
        L = summary[name]["best_layer"]
        res = json.loads((paths.RUN_DIR / "discover" / f"{name}_L{L}.json").read_text())
        sae = QwenScopeSAE.load(L)
        labels[name] = {str(f): max_activating_examples(m, sae, L, f, pool, 8)
                        for f in res["probe_top"][:5]}
    (paths.RUN_DIR / "interpret" / "top_features.json").write_text(json.dumps(labels, indent=2))
```

- [ ] **Step 3: Re-run discovery (now with interpret) and review**

Run: `.venv-sae/bin/python scripts/saemap_discover.py`
Expected: `DISCOVER DONE`; `interpret/top_features.json` exists. Read it and confirm the decision features' top snippets look cooperation/defection-related and the recog_fine features' snippets look dilemma-related (human read; per project rule do not regex-judge — eyeball or use an LLM judge if labeling formally).

- [ ] **Step 4: Commit**

```bash
git add game_theory_llm/saemap/interpret.py scripts/saemap_discover.py
git commit -m "saemap: feature interpretation via max-activating examples"
```

---

## Task 9: Causal test + mediation (`causal.py` + `saemap_causal.py`)

**Files:**
- Create: `game_theory_llm/saemap/causal.py`
- Create: `scripts/saemap_causal.py`
- Test: `tests/saemap/test_causal_hook.py`

**Interfaces:**
- Consumes: `QwenModel`, `QwenScopeSAE`, discovery shortlists, `corpus.pd_eval_set`.
- Produces (`causal.py`):
  - `add_vec_hook(model, layer, vec: torch.Tensor)` — context manager; adds `vec` ([D_MODEL]) to `model.model.layers[layer]` output residual on every forward.
  - `p_coop_steered(qm, prompt, coop_letter, layer, vec, fewshot="") -> float`.
  - `decision_sweep(qm, sae, layer, features, alphas, eval_rows, fewshot="") -> dict` — for each feature and α, mean P(coop) over eval_rows with `vec = α·unit(W_dec[:,f])`.
  - `mediation(qm, sae, layer, recog_features, eval_rows, alpha, fewshot="") -> dict` — ablate (subtract) recognition feature direction(s); report mean P(coop) shift vs baseline.
  - `random_feature_control(qm, sae, layer, alphas, eval_rows, n=5, seed=0, fewshot="") -> dict`.
- Produces (`saemap_causal.py`): `data/runs/saemap_9b/causal/{decision_sweep,mediation,controls}.json` + a printed verdict.

- [ ] **Step 1: Write the failing hook test** (`tests/saemap/test_causal_hook.py`)

```python
import torch
from game_theory_llm.saemap import causal

class _Block(torch.nn.Module):
    def forward(self, x):                      # returns a tuple like a HF decoder block
        return (x.clone(),)

class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block()])

def test_add_vec_hook_shifts_residual():
    m = _Model()
    x = torch.zeros(1, 3, 4)
    base = m.model.layers[0](x)[0]
    vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with causal.add_vec_hook(m, 0, vec):
        out = m.model.layers[0](x)[0]
    assert torch.allclose(out - base, vec.expand_as(out))
    # hook removed after context
    after = m.model.layers[0](x)[0]
    assert torch.allclose(after, base)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_causal_hook.py -v`
Expected: FAIL — no `causal` module.

- [ ] **Step 3: Write `game_theory_llm/saemap/causal.py`**

```python
"""Causal interventions: add/ablate feature directions at the SAE layer; sweeps + mediation."""
from __future__ import annotations
import contextlib, torch

@contextlib.contextmanager
def add_vec_hook(model, layer, vec):
    vec = vec.to(next(model.parameters()).dtype)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].device),) + out[1:]
        return out + vec.to(out.device)
    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()

@torch.no_grad()
def p_coop_steered(qm, prompt, coop_letter, layer, vec, fewshot=""):
    with add_vec_hook(qm.model, layer, vec):
        return qm.p_coop(prompt, coop_letter, fewshot)

def decision_sweep(qm, sae, layer, features, alphas, eval_rows, fewshot=""):
    res = {}
    for f in features:
        unit = sae.decoder_col(f, unit=True)
        res[str(f)] = {}
        for a in alphas:
            vec = a * unit
            ps = [p_coop_steered(qm, r["prompt"], r["coop_letter"], layer, vec, fewshot)
                  for r in eval_rows]
            res[str(f)][str(a)] = sum(ps) / len(ps)
    return res

def mediation(qm, sae, layer, recog_features, eval_rows, alpha, fewshot=""):
    base = [qm.p_coop(r["prompt"], r["coop_letter"], fewshot) for r in eval_rows]
    direction = torch.stack([sae.decoder_col(f, unit=True) for f in recog_features]).sum(0)
    direction = direction / direction.norm()
    abl = [p_coop_steered(qm, r["prompt"], r["coop_letter"], layer, -alpha * direction, fewshot)
           for r in eval_rows]
    return {"baseline_mean": sum(base) / len(base),
            "ablated_mean": sum(abl) / len(abl),
            "shift": sum(abl) / len(abl) - sum(base) / len(base),
            "alpha": alpha, "recog_features": list(recog_features)}

def random_feature_control(qm, sae, layer, alphas, eval_rows, n=5, seed=0, fewshot=""):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randint(0, sae.W_dec.shape[1], (n,), generator=g).tolist()
    return decision_sweep(qm, sae, layer, feats, alphas, eval_rows, fewshot)
```

- [ ] **Step 4: Run hook test to verify it passes**

Run: `.venv-sae/bin/python -m pytest tests/saemap/test_causal_hook.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `scripts/saemap_causal.py`**

```python
"""Decision dose-response sweep + recognition->decision mediation + controls."""
import json, numpy as np
from scipy import stats
from game_theory_llm.saemap.model import QwenModel
from game_theory_llm.saemap.sae import QwenScopeSAE
from game_theory_llm.saemap import causal, corpus, paths

ALPHAS = [-8, -4, 0, 4, 8]

def monotonic_trend(alphas, means):
    return float(stats.spearmanr(alphas, means).correlation)

def main():
    summary = json.loads((paths.RUN_DIR / "discover" / "summary.json").read_text())
    sanity = json.loads((paths.RUN_DIR / "sanity.json").read_text())
    fewshot = causal.__dict__.get("FEWSHOT", "")  # mirror sanity's choice if used
    fewshot = "" if not sanity.get("fewshot_used") else __import__(
        "scripts.saemap_sanity", fromlist=["FEWSHOT"]).FEWSHOT

    qm = QwenModel()
    eval_rows = corpus.pd_eval_set(limit=40)

    # decision sweep at decision-track best layer
    Ld = summary["decision"]["best_layer"]
    saed = QwenScopeSAE.load(Ld)
    dec = json.loads((paths.RUN_DIR / "discover" / f"decision_L{Ld}.json").read_text())
    top_feats = dec["probe_top"][:3]
    sweep = causal.decision_sweep(qm, saed, Ld, top_feats, ALPHAS, eval_rows, fewshot)
    ctrl = causal.random_feature_control(qm, saed, Ld, ALPHAS, eval_rows, n=5, fewshot=fewshot)
    (paths.RUN_DIR / "causal" / "decision_sweep.json").write_text(json.dumps(
        {"layer": Ld, "features": top_feats, "alphas": ALPHAS, "sweep": sweep}, indent=2))
    (paths.RUN_DIR / "causal" / "controls.json").write_text(json.dumps(
        {"layer": Ld, "alphas": ALPHAS, "random_features": ctrl}, indent=2))

    # mediation at recognition-fine best layer
    Lr = summary["recog_fine"]["best_layer"]
    saer = QwenScopeSAE.load(Lr)
    rec = json.loads((paths.RUN_DIR / "discover" / f"recog_fine_L{Lr}.json").read_text())
    med = causal.mediation(qm, saer, Lr, rec["probe_top"][:3], eval_rows, alpha=8, fewshot=fewshot)
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
git commit -m "saemap: causal decision sweep + recognition mediation + random-feature controls"
```

---

## Task 10: Results writeup + research-log entry

**Files:**
- Create: `docs/results/saemap_9b_result.md`
- Use the `/log` skill to append a research-notebook entry.

**Interfaces:**
- Consumes: `discover/summary.json`, `interpret/top_features.json`, `causal/verdict.json`, `sanity.json`.

- [ ] **Step 1: Write `docs/results/saemap_9b_result.md`**

Include: the Step-0 gate outcome (was few-shot needed; mean/std P(coop)); discovery AUCs per contrast and best layers; the named top features (decision + recog_fine) with one representative max-activating snippet each; the decision dose-response (P(coop) at each α for the top feature) vs the random-feature control; the mediation shift; and the verdict booleans. State the honest conclusion (mechanistic confirmation, partial, or null) and whether to trigger the staged 27B replication. Copy the doc to the main mirror: `cp docs/results/saemap_9b_result.md ../../docs/results/ 2>/dev/null || true` is not valid across the worktree boundary — instead `cp docs/results/saemap_9b_result.md /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/docs/results/saemap_9b_result.md`.

- [ ] **Step 2: Append a `/log` research-notebook entry** summarizing the hypothesis, method, and result.

- [ ] **Step 3: Commit**

```bash
git add docs/results/saemap_9b_result.md
git commit -m "saemap: 9B results writeup (discovery + causal + mediation)"
```

---

## Staged follow-up (NOT in this plan; gated on a positive 9B result)

Replication on `Qwen3.5-27B-Base` + `SAE-Res-Qwen3.5-27B-W80K-L0_100` via the Modal `safety` app (A100-80GB). The package is model-agnostic except `paths.py` ids, `D_MODEL=5120`, `D_SAE=81920`, `TOPK=100`, and the candidate-layer list (27B has 64 layers → sweep ~{16,24,32,40,48}). Only spin this up if `causal/verdict.json` shows `decision_causal_pass` (and ideally `mediation_pass`). This is the standing ≥2-model validation bar.

---

## Self-Review

**Spec coverage:**
- §2 decision track → Tasks 4,6,7,9. ✓
- §2 recognition track (fine+coarse) → Tasks 4,6,7. ✓
- §3 env (3.11 venv, MPS, per-layer SAE) → Task 1; Global Constraints. ✓
- §4 module layout → all modules present (sae/model/corpus/discover/interpret/causal + scripts + tests). ✓
- §5a cue-pair contrast + label-swap → Task 4 (`decision_pairs`, cues); label-swap available via `PD_SWAP` (used as a robustness control in Task 9 if desired — noted). **Gap fixed:** add label-swap to the decision sweep as an optional control in the result writeup. ✓
- §5b recognition contrasts → Task 4 `recognition_sets`. ✓
- §5c layer sweep → Tasks 6,7. ✓
- §6 P(coop) readout + intervention + mediation + controls (label-swap, random-feature, recon fidelity) → Tasks 2(recon),3(p_coop),9(sweep/mediation/random). ✓
- §7 success criteria → encoded in Task 9 verdict + Task 7 AUCs. ✓
- §8 Step-0 gate → Task 5. ✓

**Placeholder scan:** no TBD/TODO; every code step has complete runnable code. The result-writeup task describes exact fields to include (not "write something"). ✓

**Type consistency:** `QwenScopeSAE.load/encode/decoder_col(unit=)/reconstruct` consistent across Tasks 2,6,7,9; `QwenModel.residuals/residual_for_completion/p_coop` consistent across Tasks 3,5,6,8,9; `discover.diff_of_means/l1_probe/decompose_direction/top_features` consistent across Tasks 7,8,9. SAE feature tensors are `[N, D_SAE]` everywhere; residuals `[N, D_MODEL]`. ✓

**Note for executor:** the decision-track label-swap robustness check (rerun the winning feature's sweep using `PD_SWAP` prompts, expecting the effect to survive the A↔B flip) is optional within Task 9 and should be reported in Task 10 if run.
