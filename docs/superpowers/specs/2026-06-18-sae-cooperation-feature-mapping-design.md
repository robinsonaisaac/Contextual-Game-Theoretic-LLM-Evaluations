# SAE Feature-Mapping of the Cooperate↔Defect Direction — Design

**Date:** 2026-06-18
**Branch:** `feature/activation-steering` (continues the steering line of work)
**Status:** design approved; pending spec review → writing-plans

## 1. Goal & scientific claim

Our prior work fit a single "cooperation" activation-steering direction (mean-diff on the
324-story PD corpus) and showed it *causally* shifts both single decisions and live
multi-agent play. That result is real but **opaque**: it is one direction in a 4096-d space
with no account of *what* it represents.

This project tests a sharper, mechanistic claim:

> The cooperate↔defect direction decomposes into a **small set of interpretable, sparse SAE
> features**, and **clamping those features causally moves the cooperate/defect decision**.

If true, the framing result is upgraded from "a steerable direction exists" (descriptive) to
"these named features mediate the decision" (mechanistic) — the upgrade ICLR Round-1 reviewers
asked for ("descriptive not mechanistic").

We use **Qwen-Scope**, Qwen's official pretrained SAE suite (residual-stream, TopK), so we do
not train SAEs ourselves.

## 2. Scope

**In scope (first cut, fully local):**
- One model: `Qwen3.5-9B-Base` + Qwen-Scope SAE `SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`
  (32 layers, d_model 4096, d_sae 65536, L0=50, per-layer `.pt` files).
- Contrast: **cooperate vs defect** (forced commitment-cue pairs; see §5).
- Discovery (all three passes, one cached activation set): diff-of-means, L1 probe,
  steering-vector dictionary decomposition.
- Causal test: clamp discovered features, measure dose-response on P(cooperative option).

**Staged (gated on a positive 9B causal result, not built up front):**
- Replication on `Qwen3.5-27B-Base` + `SAE-Res-Qwen3.5-27B-W80K-L0_100` on cloud
  (Modal `safety` app, A100-80GB). This is our standing ≥2-model validation bar.

**Non-goals:** training SAEs; the framing (allies vs enemies) contrast; multi-agent gameplay
steering (already done with the holistic vector); touching the RLVR/Instruct models.

## 3. Environment (Step 0 prerequisite)

- No `.venv*` currently exists. System `python3` is 3.9.6 with **torch 2.8.0 + MPS working**
  and `huggingface_hub` 0.36.2, but Python 3.9 likely cannot load the `Qwen3.5` architecture.
- **Create a Python 3.11 venv** (`.venv-sae`) with: current `transformers` (Qwen3.5 support),
  `torch` (MPS), `huggingface_hub`, `safetensors`, `scikit-learn` (L1 probe), `numpy`/`pandas`.
- Memory budget verified: 9B bf16 ≈ 18 GB + one SAE layer (bf16) ≈ 1.1 GB + activations ≈ 22 GB
  peak < 34 GB **if SAEs are loaded one layer at a time** (never all 32 at once).
- Residual stream is read via `output_hidden_states=True` (`hidden_states[layer]` = post-layer
  residual, the locus Qwen-Scope SAEs hook); no custom forward hooks strictly required for
  discovery. Causal intervention does require a forward hook at the layer (see §6).

## 4. Components & module layout

New code under `game_theory_llm/saemap/` (keep units small and independently testable):

| unit | responsibility | depends on |
|---|---|---|
| `sae.py` | load a Qwen-Scope `.pt` layer; `encode(resid)->features` (TopK), `decoder_col(f)` | torch |
| `model.py` | load Qwen3.5-9B-Base on MPS; `residuals(prompts, layer)`; `p_coop(prompts)` | transformers |
| `corpus.py` | build cooperate/defect commitment-cue pairs + label-swap twin from PD JSONL | — |
| `discover.py` | diff-of-means (A), L1 logistic probe (B), steering-vector decomposition (C) | sklearn |
| `interpret.py` | max-activating-example labels for shortlisted features | model.py |
| `causal.py` | feature-clamp forward hook; coefficient sweep; random-feature + label-swap controls | model.py, sae.py |

Scripts (thin orchestrators) under `scripts/`: `saemap_setup_env.sh`, `saemap_sanity.py`
(Step-0 gate), `saemap_extract.py`, `saemap_discover.py`, `saemap_causal.py`.
Tests under `tests/saemap/` for `sae.py` (load + encode shapes, TopK L0), `corpus.py`
(pair/label-swap construction), `discover.py` (diff-of-means + probe on a synthetic set).

## 5. Contrast construction (the careful part)

A naive "append A vs B and read the pre-decision position" yields **identical** activations
(shared prefix) — no contrast. We therefore use the established contrastive-cue recipe:

- For each PD scenario, build a **commitment-cue pair**: `prompt + <cooperate cue>` vs
  `prompt + <defect cue>` (one short sentence each committing to the choice, e.g. "…decides to
  honor the agreement and cooperate." vs "…decides to break the agreement and defect.").
- Capture residuals at the **cue span** tokens, SAE-encode, **mean-pool over the span**,
  average per class. The cooperate−defect difference is the signal.
- **Label-swap control:** reuse the existing A↔B label-swap corpus so discovered features are
  the cooperate/defect axis, not the surface A/B token identity.
- Sweep candidate layers ~{8, 12, 16, 20, 24} (mid-stack, mirroring the prior layer sweep);
  pick the layer with the cleanest separation for the causal test.

## 6. Causal test

- **Readout = P(cooperative option) as the next token.** A few-shot decision prompt (2–3 worked
  PD examples ending in `Decision: A`/`Decision: B`) primes the base model to emit a letter;
  P(coop) = softmax over the {coop-letter, defect-letter} logits. Smooth dose-response signal.
- **Intervention:** forward hook at the SAE layer; for shortlisted feature f, add
  `α · unit(W_dec[:, f])` to the residual (and/or clamp the feature activation), sweeping α over
  a symmetric range. Test single top features and the small probe set together.
- **Controls (all required):**
  1. **Label-swap** — effect must survive A↔B flip.
  2. **Random-feature control** — clamping random features of matched activation magnitude must
     *not* move P(coop) (specificity, not generic perturbation).
  3. **Reconstruction fidelity** — report SAE variance-explained at the chosen layer so the
     features are trustworthy before any causal claim.

## 7. Success criteria

- **Discovery:** ≤ ~10 features separate cooperate/defect with high held-out AUC, surviving the
  label-swap control.
- **Causal:** clamping the top feature(s) produces a **monotonic, statistically significant**
  P(coop) dose-response; random-feature controls stay flat.
- **Interpretability:** top features get human-readable labels from max-activating examples
  (e.g. a "trust/partnership" feature vs a "rivalry/threat" feature).
- **Bridge (pass C):** the shortlisted features substantially reconstruct the refit holistic
  cooperation direction (high cosine), tying the sparse map to the validated steering vector.
- **Replication (staged):** the mapping reproduces on Qwen3.5-27B-Base.

## 8. Gates & risks

- **Step-0 sanity gate (hard):** before extraction, confirm 9B-Base shows a measurable,
  non-degenerate P(coop) that *responds* to scenario content. If P(coop) is saturated/degenerate,
  adjust the few-shot decision prompt before proceeding; do not invest in discovery on a flat
  readout.
- **Env:** Qwen3.5 architecture support requires current transformers on Python 3.11.
- **TopK sparsity (50 active/token):** handled by mean-pooling features over the cue span.
- **Base ≠ instruct:** the SAE is base-bound, so we cannot borrow an instruct model's cleaner
  decisions; the few-shot P(coop) readout is the mitigation and is gated by Step 0.

## 9. Artifacts & reproduce (forward-looking)

```
data/runs/saemap_9b/
    sae_cache/layer{L}.sae.pt            # downloaded Qwen-Scope layers (gitignored)
    activations/{layer}/{coop,defect}.pt # cached cue-span SAE features
    discover/{diffmeans,probe,decomp}.json
    causal/{sweep,controls}.json
    interpret/top_features.json
```

```bash
# 0. env + Step-0 sanity gate
bash scripts/saemap_setup_env.sh
.venv-sae/bin/python scripts/saemap_sanity.py        # must pass before continuing
# 1. extract cue-span SAE features (layer sweep)
.venv-sae/bin/python scripts/saemap_extract.py --layers 8,12,16,20,24
# 2. discover (A+B+C) + interpret
.venv-sae/bin/python scripts/saemap_discover.py --layer <best>
# 3. causal sweep + controls
.venv-sae/bin/python scripts/saemap_causal.py --layer <best> --features <shortlist>
```

## 10. Decisions on record (this brainstorm)

- Ambition: **discover + causally test**.
- Model: **9B-Base local first**; 27B cloud as **staged** replication (gated on 9B).
- Contrast: **cooperate vs defect** (forced commitment-cue pairs + label-swap control).
- Discovery: **all three passes** (diff-of-means + L1 probe + steering-vector decomposition).
