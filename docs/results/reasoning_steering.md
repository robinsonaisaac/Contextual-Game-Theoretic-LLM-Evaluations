# Can steering make the model better at (general) reasoning?

**Question.** The cooperation vector was behaviourally *selective* — it left
GSM8k/HumanEval/MMLU unchanged. Here we ask the opposite: can a steering vector
*improve* reasoning? We extract a **"correctness" direction** via CAA from the
model's own correct-vs-incorrect solutions and test whether adding it raises
accuracy — **validated across ≥3 benchmarks with the same vector**, because a
gain on the single benchmark a vector is fit from is task-specific overfitting,
not a capability increase.

**Answer (one line).** A correctness vector produces a **large gain on the
benchmark it is fit from** (GSM8k +15–19 pp) — but the **same vector does not
improve any other benchmark** (it degrades MMLU and GPQA accuracy-among-parsed,
and BBH is at ceiling). **By the ≥3-benchmark bar, this is not a validated
general-reasoning capability gain — it is task-specific overfitting to the
steering objective.** Steering can inflate the benchmark you tune on; it does
not transfer.

---

## Method

- **Model:** Gemma 4 E4B-it. **Position:** `mean_trace`, all 42 layers. **Best layer:** 18.
- **Vectors (CAA):** generate a CoT solution per train problem, label **correct/incorrect** (GSM8k by numeric match; MMLU by `decision == answer`), fit `v = mean(correct) − mean(incorrect)` per layer (class-balanced). Two vectors:
  - `reason_gsm8k_v1` — fit on 300 GSM8k-train (204 correct / 96 incorrect).
  - `mix_correct_v1` — fit on 150 GSM8k + 150 MMLU train items (205 / 95), to test for a more *general* direction.
- **Validation (same vector, 4 benchmarks):** held-out **GSM8k** (n=200, the fitting domain), **MMLU** (n=300), **GPQA-Diamond** (n=198), **BBH logical-deduction** (n=200). Train/eval splits disjoint. We report **accuracy-among-parsed** for MCQA benchmarks because steering shifts `<decision>`-tag compliance, which confounds raw accuracy.

## In-domain effect is real and large (and *not* a format artifact)

Held-out GSM8k (baseline 0.625, parse rate **1.000 in every condition**):

| | α=−6 | α=+3 | α=+6 |
|---|---|---|---|
| `reason_gsm8k_v1` (L18) | 0.255 | **0.780 (+15.5 pp, p=0.001)** | 0.410 |
| `mix_correct_v1` (L18) | — | 0.730 | **0.810 (+18.5 pp, p<10⁻³)** |

Clean inverted-U; the negative direction collapses accuracy (0.255), validating
the axis; steered-correct solutions are *more concise* (156 vs 178 words), so the
gain is genuine reasoning, not verbosity or tag-compliance.

## Cross-benchmark validation: it does NOT generalize

Accuracy-among-parsed at L18 (the layer where GSM8k improves), same vectors:

| benchmark | baseline | `gsm8k_v1` +3 / +6 | `mix` +3 / +6 |
|---|---|---|---|
| **GSM8k** (fit) | 0.625 | **0.780** / 0.410 | 0.730 / **0.810** |
| MMLU | 0.823 | 0.785 / 0.715 | 0.802 / 0.769 |
| GPQA | 0.600\* | 0.548 / 0.360 | — |
| BBH | 0.990† | 0.995 / 0.921 | 1.000 / 0.970 |

\* GPQA at a 512-token budget is parse-limited (baseline parse rate 0.076); its
raw accuracy *rises* with α purely from increased tag-emission while quality
falls. † BBH logical-deduction is at ceiling, so it cannot show a gain.

**Every benchmark other than the fitting one either degrades (MMLU, GPQA) or is
at ceiling (BBH).** Fitting on a *mix* of GSM8k+MMLU does not fix this — MMLU
still does not improve (0.823 → 0.802 at +3). The direction encodes
*get-this-benchmark-right*, not *reason-better-in-general*.

## Verdict

- **Does steering improve reasoning?** It improves the **specific benchmark the
  correctness contrast is derived from**, substantially (GSM8k +15–19 pp).
- **Is it a validated capability gain?** **No.** Under the ≥3-benchmark rule the
  same vector fails to improve MMLU, GPQA, or BBH. The in-fit gain is
  task-specific overfitting to the steering objective. Without cross-benchmark
  validation, the +15.5 pp GSM8k number alone would have been a misleading
  "capability" claim — exactly the failure mode the rule guards against.
- This is the same lesson as the cooperation vector's *selectivity*: CAA
  directions encode specific dispositions/skills, not a generic competence dial.

## Limitations / future work

- Single model (E4B), single best layer (18), single seed; effect sizes are from
  one run (the GSM8k contrast is p=0.001, but OOD deltas are modest).
- We did not search per-benchmark-optimal layers — that would itself be
  overfitting and would not constitute one *general* vector.
- Untried routes to a genuinely general gain: fitting on a much broader reasoning
  mixture; steering a *process* (deliberation/verification) rather than an
  outcome-correctness contrast; or the generation-structure route (using the
  framework to synthesise diverse training data for actual fine-tuning, e.g. via
  a continual-learning API) rather than inference-time steering.

## Reproduce

```bash
python3 scripts/build_reasoning_steer_corpus.py --n 300
python3 scripts/build_mixed_reason_corpus.py --n-mmlu 150 --n-gsm8k 150
python3 scripts/run_reasoning_steering.py --run-id reason_gsm8k_v1 --corpus data/runs/reason_steer/gsm8k_train.jsonl
python3 scripts/run_reasoning_steering.py --run-id mix_correct_v1   --corpus data/runs/reason_steer/mixed_train.jsonl
# validate the SAME vector across >=3 benchmarks:
for B in gsm8k mmlu gpqa bbh; do
  python3 scripts/run_reasoning_eval.py --eval $B --run-id reason_gsm8k_v1 --layers 18 --alphas="0,3,6"
done
```
