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

### Training method (RLVR) — GRPO via the Tinker cookbook
- **GRPO** (group-relative policy optimization, R1-style) using the cookbook's
  `tinker_cookbook.rl.train` loop — confirmed available. The cookbook does the RL
  (group-relative advantages from `group_size` rollouts/prompt, KL control, optim);
  **we only provide the env + verifier**, mirroring `recipes/math_rl/math_env.py`.
- **Integration (3 small pieces):**
  1. `GameTheoryEnv(ProblemEnv)` per family — `get_question()` returns the free-text
     problem; `check_answer(response)` is the exact verifier (parsed `<answer>` == game
     solution); `get_reference_answer()` returns the gold. ProblemEnv handles the
     correctness reward + a small format reward.
  2. an `RLDatasetBuilder` yielding `EnvGroupBuilder`s over a **depth curriculum** (each
     group = `group_size` rollouts of one prompt → GRPO advantage).
  3. a `train.Config(model_name=Qwen3-4B-Instruct-2507, learning_rate, groups_per_batch,
     group_size≈8, kl_penalty_coef, ...)` → `await train.main(config)`.
- Cold start: begin from the instruct model at easy curriculum depths (which already
  have reward signal); raise depth as reward saturates. No SFT warmup unless early reward
  is too sparse to learn.
- **Smoke-test first** (tiny groups_per_batch + few steps) to verify the env/reward wiring
  before the full run, as we did for SFT/eval.

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
- **Tier 2 — standard reasoning benchmarks with headroom:** **MMLU-Pro** (HF) and **BBH-Hard**
  reasoning subtasks (HF `maveriq/bigbenchhard`: multistep_arithmetic_two, web_of_lies,
  tracking_shuffled_objects_seven_objects, logical_deduction_seven_objects, geometric_shapes —
  the headroom screen keeps only non-ceiling ones; base BBH logical_deduction-3 was at ceiling).
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
2. `game_theory_llm/reasoning/gt_rl_env.py` + `scripts/tinker_grpo.py` — a thin GRPO
   wrapper: `GameTheoryEnv(ProblemEnv)` per family + an `RLDatasetBuilder` (depth
   curriculum) + a `train.Config`, calling `tinker_cookbook.rl.train.main` (mirrors
   `recipes/math_rl`). No custom policy-gradient code.
3. Eval: extend `scripts/tinker_eval.py`; add procedural generators/loaders for Dyck +
   ProntoQA/GSM-Symbolic; `scripts/headroom_screen.py`.

## Risks & mitigations
- **Verifier faithfulness** → deterministic complete templates + re-extraction gate.
- **Reward sparsity at high depth** → depth curriculum; optional tiny SFT warmup.
- **Reward hacking / lucky guesses** → wide answer ranges; report guess-baseline.
- **RLVR instability** → GRPO with KL control from the battle-tested cookbook loop (not
  hand-rolled); start with conservative LR / KL coef.
- **4B too weak for Tier-2** → headroom screen drops un-measurable benchmarks before training.
- **Cookbook env/reward wiring** → mirror `recipes/math_rl`; smoke-test on tiny
  groups_per_batch + few steps before the full run (as done for SFT/eval).

## Success criteria
A clean, defensible go/no-go on free-text game-theory RLVR transfer, plus reusable
generators + RLVR loop + headroom-screened eval suite for the full experiment.
