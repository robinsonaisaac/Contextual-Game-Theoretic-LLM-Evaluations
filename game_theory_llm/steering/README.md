# Steering pipeline

Activation-steering of cooperation/defection in Gemma 4 instruct models.
Runs on Modal A100s. Designed for the Prisoner's Dilemma corpus generated
by `game_theory_llm.generator`.

## Supported models

| short_name | HF id | gpu_tier | n_layers | candidate steering layers |
|---|---|---|---|---|
| `E2B`     | google/gemma-4-E2B-it     | small (A100-40GB) | 35 | 13, 17 |
| `E4B`     | google/gemma-4-E4B-it     | small (A100-40GB) | 42 | 16, 25 |
| `26B-A4B` | google/gemma-4-26B-A4B-it | large (A100-80GB) | 30 | 11, 16 |

Add a model by editing `configs.py`: pick a `short_name`, set `gpu_tier`,
and after a `warmup` run, fill in the actual `n_layers` (the warmup logs
the discovered architecture). `candidate_layers` defaults to a few mid-stack
guesses; refine after a real eval.

## Quick start

```bash
# 1. Build the corpus (324 PD stories) and its label-swapped twin
python3 scripts/build_steering_corpus.py \
    --src-dir data/runs/2026-05-05-sharp/stories \
    --game prisoners_dilemma \
    --out-train data/runs/2026-05-05-sharp/steering/train.jsonl \
    --out-eval  data/runs/2026-05-05-sharp/steering/eval.jsonl \
    --eval-frac 0.16 --seed 0
cat data/runs/.../train.jsonl data/runs/.../eval.jsonl \
    > data/runs/2026-05-05-sharp/steering/full_corpus.jsonl
python3 scripts/build_label_swap_corpus.py \
    --in-path  data/runs/2026-05-05-sharp/steering/full_corpus.jsonl \
    --out-path data/runs/2026-05-05-sharp/steering/full_corpus_swap.jsonl

# 2. Deploy the Modal app once (idempotent)
python3 -m modal deploy game_theory_llm/steering/modal_app.py

# 3a. FULL pipeline (extracts + fits + 14-shard eval per model). ~3-4 hr
python3 scripts/run_full_pipeline.py --model all --stage all --wait

# 3b. SPRINT mode — reuse already-fitted vectors, run a 50-story
#     held-out subset across all 3 models with both regular and
#     label-swapped corpora. ~25-30 min wall clock, ~$30-40.
python3 scripts/build_subset_corpus.py \
    --regular data/runs/2026-05-05-sharp/steering/full_corpus.jsonl \
    --swap    data/runs/2026-05-05-sharp/steering/full_corpus_swap.jsonl \
    --out-regular data/runs/2026-05-05-sharp/steering/full_corpus_subset50.jsonl \
    --out-swap    data/runs/2026-05-05-sharp/steering/full_corpus_swap_subset50.jsonl \
    --n 50 --seed 0
python3 scripts/run_subset_sprint.py    # spawns 30 shards + polls + decomposes
```

The pipeline has six stages (all idempotent):

1. `download`  — pull weights into the `safety` Modal volume
2. `warmup`    — discover architecture; verify hooks attach
3. `extract`   — generate trace + capture activations on every train story
4. `fit`       — sync bundles, mean-diff fit `vectors.pt`, push back
5. `eval`      — spawn `(layers × alphas)` shards (detached)
6. `aggregate` — pull per-shard parquets, stitch, print summary table

Each stage can be run individually:

```bash
python3 scripts/run_full_pipeline.py --model 26B-A4B --stage extract
python3 scripts/run_full_pipeline.py --model 26B-A4B --stage fit
python3 scripts/run_full_pipeline.py --model 26B-A4B --stage eval --wait
python3 scripts/run_full_pipeline.py --model 26B-A4B --stage aggregate
```

`--wait` blocks polling Modal for shard completion (~hours for full
sweeps). Without it, `eval` returns after spawning and you can `aggregate`
later once the call_ids file shows everything is done.

## State on the volume

```
/data/runs/{run_id}/                      run_id = pd_{NAME}_v1
    activations/train/*.pt                one bundle per story
    index.parquet                         metadata index
    vectors.pt                            fitted SteeringVectorSet
    shards/L{layer}_{position}_a{α}.parquet   per-cell eval results
    eval_progress.jsonl                   incremental progress (racy)
    results_sweep.parquet                 (when evaluate() is used)
```

Local mirror of these lives at `local_data/{run_id}_dl/` and
`local_data/{run_id}_results/`.

## Key findings (May 2026)

| Model | Best layer | α=−3 → α=+3 | Range |
|---|---|---|---|
| E2B   | L17 | 36% → 83% | 47pp |
| E4B   | L16 | 32% → 91% | 60pp |
| 26B-A4B | L11 | **5% → 97%** | **94pp** |

All effects highly significant (Fisher exact p < 0.001 vs α=0). The
26B-A4B's L11 mean_trace direction is the cleanest cooperation/defection
direction we've found — near-saturation at α=±2.

The L4B finding was validated by a label-swap diagnostic: swapping the
A↔B labels on the prompts (and inverting the cooperative letter) leaves
the cooperation effect mostly intact (e_coop = −42pp at α=−3, +11pp at
α=+3), with only 2-8pp of "A-letter bias" contamination. See
`scripts/build_label_swap_corpus.py`.

## Module layout

```
game_theory_llm/steering/
    configs.py         model registry (short_name -> ModelConfig)
    modal_app.py       Modal app: SteeringWorker (40GB) + SteeringWorkerLarge (80GB)
    extraction.py      generate_trace, extract_activations
    application.py     steering_hook, multi_steering_hook, generate_with_hook
    vector_fitting.py  mean-difference fit_vectors
    evaluation.py      _eval_one_cell, prune_pass, sweep_pass
    storage.py         per-bundle .pt + parquet index serialization
    models.py          dataclasses (ActivationBundle, SteeringVector, ...)
```

```
scripts/
    run_full_pipeline.py        canonical multi-model orchestrator
                                (download/warmup/extract/fit/eval/aggregate)
    run_subset_sprint.py        tight-budget sprint variant: 50-story
                                held-out subset on existing vectors,
                                regular + swap, all 3 models, 30 shards
    run_finalized_full.py       fan-out version of run_full_pipeline
                                that runs every model in parallel
    build_steering_corpus.py    PD JSONL -> steering JSONL
    build_label_swap_corpus.py  A↔B label flip diagnostic corpus
    build_subset_corpus.py      deterministic N-row subsample
    aggregate_shards.py         stand-alone parquet stitcher

    # legacy, superseded by run_full_pipeline.py but still functional:
    spawn_eval_extended_multi.py
    spawn_eval_extended.py
    spawn_eval_parallel.py
    spawn_eval.py
    check_eval.py
    check_shards.py
    run_steering.py             single-model variant
```

## Reproducibility

Every result reported in the paper section can be reproduced by
running the corresponding script with the seeds shown above:

- The 324-story PD corpus → `build_steering_corpus.py --seed 0`
- The label-swapped twin → `build_label_swap_corpus.py` (deterministic
  string substitution)
- The 50-story sprint subset → `build_subset_corpus.py --n 50 --seed 0`
- Full extracts + fits per model → `run_full_pipeline.py --model {NAME} --stage extract` then `--stage fit`
- Cross-model + swap eval (sprint mode) → `run_subset_sprint.py`

Vectors live on the Modal volume `safety` at
`/data/runs/{run_id}/vectors.pt`. The pipeline regenerates them on
demand from the activations bundle index.

The legacy spawn/check scripts still work; they're useful for ad-hoc
single-cell experiments. The orchestrator is the recommended path for
full pipelines.
