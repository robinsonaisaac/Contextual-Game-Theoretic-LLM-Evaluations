# Spec — Free-text game-theory RLVR: a lean transfer probe

**Date:** 2026-06-03
**Status:** design (awaiting review)

## Question
Can training a model with **RLVR on free-text game-theoretic scenarios** make it a
**better general long-depth reasoner** — i.e. transfer beyond the trained tasks to
held-out reasoning benchmarks? This is the *pure* test: only game-theory-derived data,
no external reasoning corpora.

Prior results motivate it: single-task SFT on *formal* game trees gave a huge in-domain
gain but **zero transfer** (and steering was likewise task-specific). Two design changes
target the likely causes: (1) **free-text presentation** so the input distribution matches
real reasoning benchmarks (prose word-problems), forcing the model to *extract structure
and reason* rather than parse a template; (2) **RLVR** so the model learns from its own
reasoning under a verifiable reward, which is the regime that produces depth/length
generalization — instead of memorizing a rigid gold CoT.

## Hypothesis
Diverse *reasoning operations* (not game flavors), presented as free text and trained by
verifiable-reward RL, instill domain-general reasoning discipline that transfers to
non-game benchmarks. Falsifiable: if held-out non-game benchmarks don't move, free-text
RLVR on narrow families still doesn't generalize.

## Scope — LEAN PROBE (this round)
Fastest credible go/no-go. NOT the full experiment (no 5-family mixture, no multi-axis
ablation, no large model). Out of scope: PPO-vs-expert-iteration comparison, RLVR-vs-SFT
ablation, scale-up models — all deferred to the full experiment if the probe is positive.

## Design

### Base model
`Qwen/Qwen3-4B-Instruct-2507` (Tinker), matching the prior SFT experiment for comparability.

### Training families (3, distinct reasoning operations, all depth-parameterized)
Chosen to render *naturally* as prose with *airtight* verifiers:
1. **Finite alternating-offer bargaining** (Rubinstein, T rounds, discount δ) — operation:
   backward induction / recursive planning. Depth = T. Verifier: closed-form SPE share.
2. **Level-k beauty contest** (p-beauty) — operation: nested belief / iterated function.
   Depth = k. Verifier: `round(start · p^k)`.
3. **Iterated dominance** in a small (2–4 action) matrix game — operation: deduction /
   elimination. Depth = elimination rounds. Verifier: IEDS solver.

### Free-text generation + verifier faithfulness
- Each problem is generated **structure-first** (sample the exact game), **solved exactly**
  (the verifier), then **rendered to prose** via deterministic, complete, entity/number-
  randomized templates with multiple surface forms per family (negotiation, market,
  contest, firms, etc.).
- **Faithfulness guarantee:** the template is a lossless encoding (all payoffs/rounds/rules
  stated). Optional LLM paraphrase for richness is **gated by a re-extraction check**
  (re-parse the structure from the prose; keep only if it reproduces the solver's answer).
- Prompt asks for reasoning then `<answer>VALUE</answer>`. **Reward-hacking guard:** widen
  answer ranges so blind guessing is rare (track guess-baseline per family).

### Training method (RLVR)
- **Curriculum expert-iteration (rejection-sampling RLVR)** for the probe: per prompt,
  sample K rollouts (Tinker sampling client) at a curriculum depth; **reward = verifier
  0/1**; SFT on the rewarded (correct) rollouts; iterate, raising depth as accuracy
  saturates. This is the leanest robust verifiable-reward method, reuses the working
  `tinker_sft.py`, and avoids PPO instability for a first read.
- **Implementation check:** if `tinker_cookbook` provides a turnkey policy-gradient
  (PPO/GRPO) RLVR recipe, prefer it (closer to "true" RLVR); otherwise use expert-iteration.
  Either way the reward is the game verifier. (Decide at implementation; note in results.)
- Cold start: begin from the instruct model at easy depths (which already have reward
  signal); add a tiny format-SFT warmup only if early reward is too sparse.

### Evaluation — headroom-screened, depth-scaled, tiered (~4 benchmarks)
- **Step 0 — headroom screen (do FIRST):** run base Qwen3-4B on all candidates; **keep
  only those at 25–80% accuracy** (this is non-negotiable — BBH was at ceiling, GPQA
  parse-broken at floor; both were useless validators).
- **Tier 0 — in-domain held-out:** the *held-out family* (train on 2 of the 3, test on the
  3rd) + **depth extrapolation** (test deeper than trained). The direct long-depth claim.
- **Tier 1 — depth-scaled, non-game, verifiable (the key diagnostics), both procedurally
  generated (no dataset-availability risk):** **Dyck-language nesting** (nesting depth =
  structural analog of game depth) + **ProntoQA-style synthetic proofs** (accuracy vs
  proof-hop depth). Both are self-built generators with exact verifiers.
- **Tier 2 — one standard naturalistic benchmark with headroom:** **MMLU-Pro** (HF dataset;
  reasoning-heavy, unlike base MMLU). Dropped automatically if the headroom screen shows
  ceiling/floor.
- **No-regression check:** GSM8k (base ≈0.90) must not drop.

### Metrics
Per-benchmark accuracy and **accuracy-among-parsed** (steering taught us raw accuracy is
confounded by format/parse shifts); **transfer-vs-depth curves** for the depth-scaled
benchmarks (no single aggregate hiding per-type failure); guess-baseline per family.

### Probe read (go/no-go)
**Positive (scale to full experiment):** the RLVR checkpoint improves the **held-out family
and/or depth-extrapolation** AND lifts **≥1 non-game Tier-1/2 benchmark** (accuracy-among-
parsed), with **no GSM8k regression**.
**Negative:** in-domain improves but no non-game benchmark moves → free-text RLVR on narrow
families still doesn't generalize (clean, informative negative).

## Components to build
1. `game_theory_llm/reasoning/freetext.py` — generators + exact verifiers + prose templates
   for the 3 families (unit-tested vs solvers; faithfulness re-extraction check).
2. `scripts/tinker_rlvr.py` — curriculum expert-iteration loop (sample → verify → SFT →
   iterate), reusing the Tinker SFT/sampling clients.
3. Eval: extend `scripts/tinker_eval.py`; add procedural generators/loaders for Dyck +
   ProntoQA/GSM-Symbolic; `scripts/headroom_screen.py`.

## Risks & mitigations
- **Verifier faithfulness** → deterministic complete templates + re-extraction gate.
- **Reward sparsity at high depth** → depth curriculum; optional tiny SFT warmup.
- **Reward hacking / lucky guesses** → wide answer ranges; report guess-baseline.
- **RLVR instability** → expert-iteration (no policy-gradient) for the probe.
- **4B too weak for Tier-2** → headroom screen drops un-measurable benchmarks before training.
- **Tinker RL API uncertainty** → smoke-test the loop on a tiny scale before the full run
  (as done for SFT/eval).

## Success criteria
A clean, defensible go/no-go on free-text game-theory RLVR transfer, plus reusable
generators + RLVR loop + headroom-screened eval suite for the full experiment.
