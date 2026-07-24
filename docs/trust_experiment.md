# Trust-vector experiment — runbook

A second steering direction (Trust vs Don't-Trust) extracted from a
*one-sided* trust-game corpus rather than the symmetric PD corpus. The
load-bearing diagnostic is the cosine similarity between the trust
vector and the cooperation vector: if they're the same direction with a
different label, this is a relabelling; if they're meaningfully distinct,
we have a second axis we can steer along.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Payoff cell registry (`game_theory_llm/trust_games.py`) | ✅ done; 10 cells, p* ∈ [0.02, 0.95] |
| 2 | Generator script (`scripts/generate_trust_stories.py`) | ✅ code in repo; ~$5-10 to run |
| 3 | Steering JSONL converter (`scripts/build_trust_steering_corpus.py`) | ✅ done |
| 4 | Extract + fit on Modal | ⏳ ready to run via `run_full_pipeline.py` |
| 5 | Cosine + calibration analysis (`scripts/analyze_trust_vs_coop.py`) | ✅ done |

## Payoff cell design

10 cells span the break-even betrayal probability $p^\* = (R-P)/(R-S)$
across the unit interval (where a risk-neutral EV-maximiser flips from
trust to no-trust):

| cell_id | R | P | S | $p^*$ |
|---|---|---|---|---|
| catastrophic_betrayal | 100 | 80 | −1000 | 0.018 |
| asymmetric_loss | 100 | 80 | −100 | 0.10 |
| high_downside | 100 | 80 | 0 | 0.20 |
| moderate_downside | 100 | 70 | 0 | 0.30 |
| high_stakes_balanced | 200 | 120 | 0 | 0.40 |
| classic_trust_game | 100 | 50 | 0 | 0.50 |
| slight_upside | 100 | 40 | 0 | 0.60 |
| high_upside | 100 | 25 | 0 | 0.75 |
| low_downside_high_upside | 100 | 15 | 0 | 0.85 |
| freebie_trust | 100 | 5 | 0 | 0.95 |

This lets us read off a *trust calibration curve*: trust rate as a
function of $p^*$. A risk-neutral Bayesian model with prior $\bar p$ on
betrayal trusts iff $\bar p < p^*$, so the calibration curve should be a
step function at $p^* = \bar p$. Real models will have softer curves; the
slope and inflection both characterise the model.

## End-to-end run

Set environment (OpenRouter API key needed for step 2):

```bash
export OPENROUTER_API_KEY=...
```

```bash
# 1. Generate trust vignettes. 10 cells × 5 framings × 6 stories = 300 total,
#    ~$5-10 at Claude Opus 4.7 prices.
python3 scripts/generate_trust_stories.py \
    --out-dir data/runs/trust_v1/stories \
    --model opus --n-per-cell 6 --concurrency 10 --seed 0

# 2. Convert to the steering JSONL format
python3 scripts/build_trust_steering_corpus.py \
    --src-dir data/runs/trust_v1/stories \
    --out-train data/runs/trust_v1/steering/train.jsonl \
    --out-eval  data/runs/trust_v1/steering/eval.jsonl \
    --out-full  data/runs/trust_v1/steering/full_corpus.jsonl \
    --eval-frac 0.16 --seed 0

# 3. (Optional) build the label-swap diagnostic corpus
python3 scripts/build_label_swap_corpus.py \
    --in-path  data/runs/trust_v1/steering/full_corpus.jsonl \
    --out-path data/runs/trust_v1/steering/full_corpus_swap.jsonl

# 4. Deploy the Modal app (idempotent)
python3 -m modal deploy game_theory_llm/steering/modal_app.py

# 5. Extract + fit on each model. Reuse the registry; just override run_id.
#    ~70 min per model + ~3 min fit. Use a distinct run_tag.
python3 scripts/run_full_pipeline.py --model E4B --stage extract --run-id pd_E4B_trust_v1 \
    --train-stories data/runs/trust_v1/steering/train.jsonl --wait
python3 scripts/run_full_pipeline.py --model E4B --stage fit --run-id pd_E4B_trust_v1

# 6. Run the eval sweep — use the same layers we used on PD; the cosine
#    check tells us whether to refine.
python3 scripts/run_full_pipeline.py --model E4B --stage eval --run-id pd_E4B_trust_v1 \
    --eval-stories data/runs/trust_v1/steering/full_corpus.jsonl --wait
python3 scripts/run_full_pipeline.py --model E4B --stage aggregate --run-id pd_E4B_trust_v1

# 7. The critical diagnostic
python3 scripts/analyze_trust_vs_coop.py \
    --coop-vectors local_data/pd_full_v1_dl/pd_full_v1/vectors.pt \
    --trust-vectors local_data/pd_E4B_trust_v1_dl/pd_E4B_trust_v1/vectors.pt \
    --trust-results local_data/pd_E4B_trust_v1_results/results_full.parquet \
    --out-dir local_data/trust_vs_coop_analysis
```

## What to look for

Three outcomes from step 7 and what each tells us:

1. **Peak cosine > 0.8**: trust and cooperation are essentially the same
   internal direction in Gemma 4. The PD-cooperation vector already
   captures most of what we'd call "trust." We can stop and write up the
   negative result — interesting in its own right because it argues the
   model has a single prosocial axis rather than separate trust /
   cooperation primitives.
2. **Peak cosine 0.5–0.8**: meaningfully overlapping. Trust-steering and
   cooperation-steering are correlated but distinct. Worth running the
   transfer experiments described in `game_play_harness.md` to see if
   one transfers to social-deduction games better than the other.
3. **Peak cosine < 0.5**: trust is a genuinely separate direction. The
   strongest version of the paper's safety claim: the same procedural
   methodology yields multiple, controllable, mechanistically distinct
   behavioural axes.

In all three cases the **trust calibration curve** is a publishable
artifact: it characterises the model's natural trust threshold without
any steering. If the model is a perfect Bayesian with prior $\bar p$ on
betrayal, the curve is a sigmoid centred at $\bar p$; we report both
$\bar p$ and the slope.

## Replicating across models

Repeat steps 5–7 with `--model E2B` and `--model 26B-A4B`. The cross-
model story we want to be able to tell:

- Does cosine grow or shrink with model scale? (i.e. do bigger models
  develop a *separate* trust direction or do they collapse trust into
  cooperation?)
- Does the calibration curve get sharper / closer to perfect-Bayes with
  scale?

These are direct cross-model claims that strengthen the paper without
new game-playing infrastructure.
