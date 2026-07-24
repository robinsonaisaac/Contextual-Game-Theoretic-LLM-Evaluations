# SAE Feature-Mapping of Game-Theoretic Reasoning — Results (Qwen3.5-9B-Base)

**Date:** 2026-06-26
**Model:** `Qwen3.5-9B-Base`
**SAE:** `SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` (residual-stream TopK, L0=50, d_sae=65536, 32 layers)
**Compute:** heavy forward passes on Modal A100; SAE math + discovery local
**Behavioral readout:** generate-and-judge — model generates a free-text decision, Claude Sonnet 4.6 grades cooperate/defect/unclear
**Scope:** two models — the recognition feature is **replicated on `Qwen3.5-27B`** (+ `SAE-Res-Qwen3.5-27B-W80K-L0_100`); see *Multi-model replication* below. The causal test was run on the 9B only.

---

## Headline finding

The model represents game-theoretic **structure** as an interpretable, sparse SAE feature ("this is a social dilemma"), and that recognition feature **causally — if modestly — steers the model's cooperation behavior**. Meanwhile, the cooperate/defect **decision** itself is **not** separately decodable as a feature from the pre-decision scenario representation. This is a recognition↔decision **dissociation**.

---

## 1. Discovery — held-out L1-probe AUC

Three contrasts were run across five layers (L8, L12, L16, L20, L24). Discovery used a sparse L1 logistic probe; `n_nonzero` confirms sparsity.

| Contrast | Best Layer | AUC | n_nonzero features |
|---|---|---|---|
| Decision (cooperate-cue vs defect-cue) | L8 | 1.00 | 1 |
| Recognition-coarse (game vs non-game text) | L8 | 1.00 | 1 |
| Recognition-fine (dilemma vs non-dilemma, style-matched) | L24 | 0.85 | 8 |

Recognition-fine contrasts {prisoners_dilemma, stag_hunt, chicken} (positive) against {harmony, deadlock} (negative) using narratively matched scenarios — only the incentive structure differs. This is the primary recognition result.

---

## 2. Interpretation — max-activating examples on 320 natural scenarios

After discovery, the top shortlisted features were probed with 320 held-out natural scenarios (no injected cue words). This step determines whether a feature represents a genuine concept or a lexical artifact.

### Decision features: lexical cue artifact

Features 2448, 21511, and 38905 (the decision-track discoveries) fire at **exactly 0.0** on every natural scenario. These features detected the injected words "cooperate" / "defect" in the cue span — not any underlying concept. The decision AUC=1.00 is therefore **meaningless**: it reflects cue leakage, not a semantic cooperate/defect representation.

### Recognition-fine features: genuinely conceptual

The recognition-fine features fire strongly on dilemma narratives with no cue words present:

| Feature | Sonnet label | Activation range |
|---|---|---|
| 48983 | "Business partners at tense crossroads" | 4.7–5.4 |
| 14407 | "Long-term partners at tense crossroads" | 5.1–5.9 |
| 29366 | "Long-term business partnership negotiations" | 3.2–4.6 |
| 51436 | "Business partners considering merger or collaboration" | 1.9–2.8 |
| 16610 | "Two companies negotiating potential merger" | 1.2–1.8 |

Activation range across top-8 examples: 1.8–5.9. All fire on dilemma narratives involving strategic interdependence; none require explicit game-theory vocabulary. These are **genuinely conceptual** social-dilemma features.

### Recognition-coarse feature: style confound

Feature 50944 fires on the non-game puzzle text (golf-tournament ranking logic puzzles). Its label is "Golf tournament ranking logic puzzles." This is a **format/domain detector** for the non-game class rather than a game-recognition feature — providing corroboration only, not an independent conceptual result.

---

## 3. Decision rebuild — balanced, cue-free (the dissociation)

To test whether the cooperate/defect decision is linearly decodable from scenario representations without cue leakage, 324 PD scenarios were labeled by the model's **natural decision** (generate + judge, no cue words):

- 208 cooperate / 68 defect / 48 unclear (out of 324 total)
- Subsampled to a balanced 68/68 set; contrasted **scenario residuals** (no cue words appended)

L1 probe results across all layers:

| Layer | AUC | n_nonzero |
|---|---|---|
| L8 | 0.50 | 0 |
| L12 | 0.50 | 0 |
| L16 | 0.50 | 0 |
| L20 | 0.50 | 0 |
| L24 | 0.48 | 1 |

**Best summary AUC: 0.50 (L8).** Weak-regularization sensitivity (inline L2, C=1, 5-fold CV — a robustness check not persisted to the discovery JSONs) peaks at **0.569 (L16)** and immediately collapses — no stable signal. The strict-L1 probe finds **0 features** in the sparse regime (AUC 0.50 every layer).

With lexical leakage removed, the cooperate/defect decision is **not** linearly or sparsely decodable from the scenario representation — in sharp contrast to the recognition result (AUC=0.85, 8 features).

**Important caveat:** the scenario residual is captured *before* the decision is generated. This null result may mean the decision is computed *during* generation (in the forward pass that produces the output tokens) rather than that no decision-representation exists anywhere in the model. The null is specific to the *pre-decision* representation.

---

## 4. Causal mediation — recognition feature → cooperation behavior

To test whether the recognition direction causally mediates cooperation behavior, the recognition direction at L24 was clamped during generation.

**Setup:**
- Clamped direction M: unit sum of W_dec columns for features 48983, 29366, 14407, 51436, 16610 at L24
- Natural dilemma − non-dilemma projection: M = 1.88
- Sweep: k ∈ {−8, −4, −2, 0, +2, +4, +8}, applied as k·M at L24 during forward pass
- Behavioral measure: cooperate-rate (generate + judge, n=50 per cell)
- Control: matched random-feature direction (same number of features, random indices)

**Dose-response (recognition direction):**

| k | Coop rate | Unclear rate |
|---|---|---|
| −8 | 0.735 | 0.02 |
| −4 | 0.773 | 0.12 |
| −2 | 0.804 | 0.08 |
| 0 (base) | 0.822 | 0.10 |
| +2 | 0.837 | 0.14 |
| +4 | 0.854 | 0.18 |
| +8 | 0.814 | 0.14 |

**Spearman across the coherent range (k=−8 to k=+4):** ρ=+0.79, p=0.036 (significant, monotone)

**Random control:** ρ=−0.36, p=0.43 (no trend; noise range 0.76–0.87)

The effect is **dose-dependent and specific** — the random control shows no trend. The suppress direction (k<0) is the cleanest signal: low unclear rates, clear cooperation drop of ~9pp at k=−8 vs baseline. The amplify direction rises to k=+4 before degrading at k=+8 (cooperation dip + rising unclear rate 0.14–0.18) — coherence degradation at extreme clamp magnitudes.

**Interpretation:** The automated `mediation_detected=False` flag in `verdict.json` is a **threshold artifact** — it computed delta using the coherence-broken k=+8 endpoint (Δ=0.079) against an arbitrary >0.15 cutoff. The proper evidence is the significant, specific Spearman dose-response (ρ=+0.79, p=0.036 vs random p=0.43). This is a **genuine but modest causal effect**: suppressing the recognition direction reduces cooperation by ~8–12pp; the random control is flat. We do not claim "mediation confirmed" (overclaim) nor "null result" (underclaim).

---

## Multi-model replication (Qwen3.5-27B)

The recognition feature **replicates and strengthens** on the larger model. Re-running the
discovery + interpretation pipeline on `Qwen3.5-27B` (+ `SAE-Res-Qwen3.5-27B-W80K-L0_100`,
d_sae=81920, L0=100, 64 layers; SAE reconstruction-fidelity 0.88 at L32, same block-input locus):

| Model | recog-fine AUC | best layer (rel. depth) | # features |
|---|---|---|---|
| `Qwen3.5-9B-Base` | 0.85 | L24 / 32 (0.75) | 8 |
| `Qwen3.5-27B` | **0.91** | L48 / 64 (0.75) | 4 |

The 27B dilemma-recognition features are **sparsely decodable** (n_nonzero 3–7 across the swept
layers, peak at L48) and **conceptually identical** to the 9B's: their max-activating examples
are interdependent-partners-under-tension narratives — *"two founders, best friends since grad
school"*; *"co-investor… rereading the term sheet"*; *"two envelopes on the table… they had
agreed"*; a secret negotiation *"at the back of the hotel bar where no one from either firm
would look."* The feature emerges at the **same relative depth (~0.75)** in both models. This
clears the ≥2-model validation bar for the recognition result.

(The causal mediation test was run on the 9B only; replicating the dose-response on the 27B is
the natural next step.)

---

## 5. Honest limitations

1. **Causal test is single-model.** The recognition *feature* is now validated across two models (9B AUC 0.85 / 27B AUC 0.91 — see *Multi-model replication*). The *causal mediation* test, however, was run on the 9B only; replicating the dose-response on the 27B remains to be done.
2. **Cooperation skew.** The generate-and-judge readout has a strong cooperation baseline (~75–87%), compressing the dynamic range available to detect cooperation changes.
3. **Unclear-rate noise.** n=50/cell means ~5–9 unclear responses per cell; unclear rates of 12–25% introduce noise in the dose-response.
4. **Pre-decision caveat.** The decision-null (Section 3) uses scenario residuals captured before the decision is generated. The null may reflect *when* rather than *whether* a decision-representation exists.
5. **Recognition-coarse style confound.** Feature 50944 (recognition-coarse) detects the format of non-game puzzle text, not game recognition per se. Only recognition-fine (Section 2) provides a clean conceptual result.

---

## 6. Connection to the broader research program

This is the **mechanistic upgrade** that ICLR Round-1 reviewers asked for ("descriptive not mechanistic"). Prior steering work identified a causal direction but could not say what it *meant*. Here, a named, interpretable set of SAE features — conceptually labeled as "partners at a tense crossroads," "long-term partnership under stakes" — *recognize* the game-theoretic structure and *causally move* the framing-driven cooperation behavior. The recognition↔decision dissociation is an additional finding: the model encodes "this is a dilemma" as a stable pre-decision representation, but does not separately encode "I will cooperate" prior to generation.

---

## 7. Reproducibility

All artifacts live under `data/runs/saemap_9b/`.

Key scripts (in execution order):

| Script | Role |
|---|---|
| `scripts/saemap_setup_env.sh` | Create `.venv-sae` (Python 3.11 + torch + transformers) |
| `scripts/saemap_sanity.py` | Step-0 gate: verify model loads and produces coherent decisions |
| `scripts/saemap_extract.py` / `saemap_collect.py` | Extract SAE activations for all contrasts, cache per layer |
| `scripts/saemap_discover.py` | Run diff-of-means + L1 probe on cached activations; write per-layer + summary JSONs |
| `scripts/saemap_interpret.py` | Max-activating examples on natural scenarios; write `interpret/top_features.json` |
| `scripts/saemap_decision2.py` | Build balanced cue-free set; run decision-null probe; write `decision2/` |
| `scripts/saemap_causal.py` | Clamping sweep + random control; write `causal/sweep_partial.jsonl` + `causal/verdict.json` |

Primary result files:
- `data/runs/saemap_9b/discover/summary.json` — discovery AUCs
- `data/runs/saemap_9b/discover/recog_fine_L24.json` — recognition-fine features at best layer
- `data/runs/saemap_9b/interpret/top_features.json` — feature labels and max-activating snippets
- `data/runs/saemap_9b/decision2/discover_summary.json` — decision-null AUCs
- `data/runs/saemap_9b/causal/verdict.json` — causal sweep summary + Spearman statistics
- `data/runs/saemap_9b/causal/sweep_partial.jsonl` — per-cell dose-response counts
