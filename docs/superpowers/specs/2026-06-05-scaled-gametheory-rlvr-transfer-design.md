# Spec — Scaled game-theory RLVR: does it transfer to *general* reasoning, and why?

**Date:** 2026-06-05  **Status:** design (awaiting decisions)

## What we already know (constraints this design must respect)
- **Within-operation generalization works** (RLVR on free-text game theory: depth-extrapolation
  +12.5 pp, p≈0.02, replicated; generalizes across depth *and* family). Not re-tested here.
- **Cross-operation transfer to general benchmarks is ~+3 pp, within noise** in both the lean and
  scaled runs (MMLU-Pro +2, BBH-Hard +4; consistent positive *direction*, never significant).
- **Implication (power):** detecting a true +3 pp effect at 80% power needs **n≈4,400/arm**; +5 pp
  needs ~1,570; +10 pp needs ~390. Naive "scale the training" without addressing power → another
  null. So this experiment is equally about **making the effect bigger, measuring it with power,
  and attributing it.**

## Precise question
Does game-theory RLVR transfer to *general* reasoning, and which mechanism governs it?
Three hypotheses, each predicting a different path to a generally-better reasoner:
- **H_op (islands):** transfer is operation-specific. Predicts: a benchmark improves only to the
  extent its required reasoning *operations* were trained. Path to general → cover every operation.
- **H_gen (discipline):** RLVR instills general reasoning discipline (decompose, verify, persist,
  longer chains) that helps all reasoning. Predicts: roughly uniform transfer regardless of operation
  overlap. Path to general → scale steps.
- **H_scale (threshold):** transfer only emerges above a base-capability threshold. Predicts: null at
  4B, positive at 8–30B. Path to general → bigger base.

## Decision-grade readouts (designed for power + attribution)
1. **Operation-overlap regression (PRIMARY, mechanism).** Label every eval *item* with the reasoning
   operation(s) it requires; regress per-item correctness improvement on `trained_operation_overlap`
   (+ depth + benchmark fixed effects), pooled across *all* eval items. This pools power across
   thousands of items and directly separates H_op (significant positive slope) from H_gen (flat slope,
   uniform intercept lift) — and can be significant even when individual benchmark means are not.
2. **Operation-isolated transfer matrix (PRIMARY, the decisive cheap test).** Train on operation-set A;
   evaluate on **operation-isolated synthetic benchmarks** (each stresses ONE operation), some in A,
   some held out. A-matched move but held-out don't ⇒ H_op. Both move ⇒ H_gen.
3. **Powered per-benchmark Δ (SECONDARY, effect size).** ≥5 naturalistic benchmarks, n powered for a
   ≥5 pp effect (~1,500/arm), accuracy-among-parsed, with Holm correction.
4. **Scale curve (SECONDARY, threshold).** Run the headline config at 4B / 8B / 30B-A3B; plot transfer
   vs base capability to test H_scale (critical-threshold style).
5. **Depth-extrapolation (replication anchor).** Confirm the in-domain gain replicates at scale.

## Design

### A. The verifiable family library (~10 reasoning operations)
Structure-first generator + exact verifier + free-text renderer + depth dial + **operation label**.
Target coverage of the operations that general benchmarks require:

| family | reasoning operation | depth dial | verifier |
|---|---|---|---|
| alternating-offer bargaining / centipede | backward induction / planning | rounds | closed-form SPE |
| Cournot / iterated dominance | deduction / elimination | elim. rounds | IESDS solver |
| level-k beauty contest | nested belief / ToM | k | round(start·pᵏ) |
| subtraction / Nim (Sprague-Grundy) | combinatorial / modular | pile size | XOR / mod |
| pure-strategy NE in N×N matrix | constraint satisfaction / fixed point | matrix size | NE solver |
| 2nd-price / first-price auction optimal bid | counterfactual / expected value | #bidders | analytic optimum |
| 3-player coalition / Shapley value | combinatorial counting / aggregation | #players | Shapley solver |
| sequential lookahead (small game tree, prose) | recursion / multi-step lookahead | plies | minimax |
| signaling / Bayesian posterior update | probabilistic reasoning | #signals | Bayes solver |
| repeated-game grim-trigger threshold | recursion / discounting | — | closed-form δ* |

Each item: prose problem → `<answer>VALUE>`; **faithfulness re-extraction gate** (re-parse structure
from the prose; keep only if it reproduces the solver). Reward-hacking guard: wide integer answer
ranges; report per-family guess-baseline. Reuse the existing 4 families; add ~6.

### B. Eval suite (operation-labeled, headroom-screened, powered)
- **Operation-isolated synthetic, depth-scaled** (for readout #2): Dyck (recursion), ProntoQA
  (deduction-depth), Countdown/24 (arithmetic-search), ZebraLogic (constraint-satisfaction),
  tracking-objects (state-tracking). Self-generated so we control difficulty into the measurable band
  per model (avoids the 4B Dyck-floor / ProntoQA-ceiling problem).
- **Naturalistic, powered** (readout #3): GSM-Symbolic + MATH-500 (arithmetic/algebra), MMLU-Pro
  (mixed), BBH-Hard per-subtask (each operation-labeled), MuSR (multi-step). n powered for ≥5 pp.
- **No-regression:** GSM8k.
- **Mandatory headroom screen first** (keep 25–80% at each model scale); operation-label every item.

### C. Models / scale curve
Qwen3-4B-Instruct → Qwen3-8B → Qwen3-30B-A3B (MoE, ~3B active = efficient stronger base). All Tinker-hosted.

### D. GRPO config
Cookbook `rl.train`, group_size 16, depth curriculum (easy→hard), KL anchor (kept the GSM8k
regression away), **hundreds** of steps (not 30), **2–3 seeds** per headline cell (RLVR is
high-variance — seeds give the CI on the *training* effect itself). Fixed HPs across families
(cross-dataset HP fixation) for fair comparison.

### E. Iso-FLOP allocation (how to spend the budget)
Under a fixed Tinker budget the three knobs are **base size S**, **steps T**, **operation coverage D**.
Run a small iso-FLOP set (same total FLOPs, different S/T/D mixes) to learn which axis buys the most
transfer — so the big spend goes to the right place rather than guessed.

## Phases & go/no-go gates (cost-phased — do NOT run later phases on a null)
- **Phase A — build family library** (~6 new families + verifiers + faithfulness gate; unit-tested). No GPU.
- **Phase B — calibration & headroom** (base models on all families + all benchmarks; operation-label
  items; set per-scale difficulty bands). **Gate 1:** ≥4 measurable benchmarks with headroom at ≥1 scale
  AND learnable families, else redesign difficulty.
- **Phase C — operation-overlap pilot** (the decisive, relatively cheap experiment): one GRPO run on
  operation-set A at the best affordable scale; eval the operation-isolated matrix + the overlap
  regression. **Gate 2:** significant overlap slope (H_op) OR significant uniform lift (H_gen) OR
  neither. This *alone* answers the mechanism question and the "is broad coverage worth it" question.
- **Phase D — scale curve + full coverage** (only if C is non-null): all ~10 families, 4B/8B/30B-A3B,
  many steps, seeds; powered naturalistic eval. Produces the headline transfer number with CIs.
- **Phase E — attribution ablations** (multi-axis, one factor at a time): coverage (single vs mixture),
  RLVR-vs-SFT-on-identical-data (does the RL signal matter or just the data?), counterfactual
  trivial-but-same-format control (isolates reasoning from format/RL), KL on/off, iso-FLOP points.
- **Phase F — scale the winner + write-up.**

## Statistical plan
- **Power:** target detectable effect ≥5 pp ⇒ n≈1,500/arm on the ~5 primary naturalistic benchmarks;
  the operation-overlap regression pools all items for additional power on the mechanism.
- **Seeds:** 2–3 GRPO seeds per headline cell; report effect mean ± between-seed sd.
- **Multiple comparisons:** Holm across benchmarks; pre-register the overlap regression as primary.
- **Metrics:** accuracy-among-parsed (raw is confounded by format shifts — established); per-operation,
  per-depth breakdowns (no single hidden aggregate); guess-baseline per family.
- **Pre-registration:** fix hypotheses, primary readouts, and the keep-set before the big spend.

## Risks & mitigations
- **Effect genuinely <5 pp** → not a useful "general reasoner" gain anyway; the overlap regression
  still yields a publishable mechanism result, and we stop at Phase C/D rather than overspend.
- **4B near ceiling** → the scale curve directly tests this (H_scale); 30B-A3B gives a stronger base cheaply.
- **Cost blow-up** → strict gates; Phase C is the cheap decisive test; iso-FLOP picks the efficient axis.
- **Verifier faithfulness / reward hacking** → re-extraction gate, wide ranges, guess-baselines.
- **Benchmark contamination** → procedural synthetic benchmarks; fresh seeds.

## Cost tiers (for the decision)
- **Tier 1 — mechanism (recommended first):** Phase A + B + C at one scale (8B or 30B-A3B) + the
  operation-overlap analysis. Answers H_op vs H_gen and "is broad coverage the path." Moderate cost.
- **Tier 2 — full:** + Phase D scale curve + full-coverage powered run (2–3 seeds). The headline
  transfer result. Large cost.
- **Tier 3 — complete:** + Phase E ablations + iso-FLOP + a second model family (Llama) for robustness.
  Publication-grade. Largest cost.

## Decisions needed
1. **Budget tier** (1 / 2 / 3) — controls how much of D/E we run.
2. **Primary base scale** for the Phase-C decisive test (8B vs 30B-A3B).
3. **Which hypothesis to privilege** if forced to choose (mechanism via operation-overlap = my
   recommendation; it's the cheapest high-information experiment and dictates everything downstream).
