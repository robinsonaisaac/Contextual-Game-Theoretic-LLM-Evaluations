# Experiments: reproduction index

Every experiment in the paper, mapped to the script that reproduces it, its
inputs, its outputs, and the paper artifact it backs. Each row is runnable from
the repo root with `python3 scripts/<script>`.

## Prerequisites

- **Python**: `python3` (system Python 3.9). Install deps: `pip install -e .` (extras: `.[steering]`).
- **LLM judge / story generation**: needs `OPENROUTER_API_KEY`. The `.env` lives in the **main repo root** (gitignored); in a worktree, source it explicitly:
  `set -a; source /path/to/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a`.
- **GPU steering/eval**: runs on Modal app **`safety`** (volume `safety`). Auth via `~/.modal.toml`. Deploy/refresh with
  `python3 -m modal deploy game_theory_llm/steering/modal_app.py`.
- **Tests**: `python3 -m pytest tests/ -q`.

## Steering run-id / vector registry

The fitted steering vectors live on the Modal volume at `runs/<run_id>/vectors.pt`,
keyed by `(layer, position)`. All headline steering uses `mean_trace`.

| run_id | model | best cell | corpus | role |
|---|---|---|---|---|
| `pd_full_v1` | Gemma 4 E4B-it (4B) | L16 `mean_trace` | 324 PD vignettes | **cooperation vector** (headline) |
| `pd_E2B_v1` | Gemma 4 E2B-it (2.3B) | L17 `mean_trace` | PD vignettes | cooperation, small |
| `pd_26B_A4B_v1` | Gemma 4 26B-A4B (MoE) | L11 `mean_trace` | PD vignettes | cooperation, large |
| `pd_E4B_trust_v1` | Gemma 4 E4B-it | L16 `mean_trace` | 300 trust-game vignettes | **trust vector** (control) |

Model configs (candidate layers per model) are in `game_theory_llm/steering/configs.py` (`MODEL_CONFIGS`, `run_id_for()`).

## Data: committed vs regenerable

- **Committed** (reproduce without the volume): all eval corpora under `data/runs/*/`*_eval.jsonl* (moral, reasoning, capability), the PD steering corpus (`data/runs/2026-05-05-sharp/steering/`), and small results JSON (`data/runs/{game_steering_v1,sh_steering_v1}/results/`).
- **Gitignored / regenerable**: `local_data/` (volume syncs, `.pt` activation bundles), generated trust stories, bulky game match logs (`data/runs/*/seed*.jsonl`).

---

## 1. Behavioral sweep (7 models × 6 games × 5 framings)
- **Paper**: §3.1–3.2, Table 1–2, Figs 1–2, App contrast stats.
- **Reproduce**: `python3 scripts/run_production.py` (generation) → analysis in `game_theory_llm/analysis/`.
- **Output**: `analysis_output/`, figures in the paper's `ResultsFigures/`.

## 2. Mechanistic probing (layer-wise probes + RSA)
- **Paper**: §4 (mechanistic_analysis.tex), Fig 3 (RSA), App layer-probe.
- **Corpus**: `python3 scripts/build_mech_interp_corpus.py`.
- **Run**: extraction via `run_full_pipeline.py --stage extract`; probing in `game_theory_llm/steering/probing.py` (`run_full_probe_analysis`). Smoke test: `scripts/test_probing.py`.

## 3. PD cooperation dose-response (3 Gemma variants)
- **Paper**: §5.3, Fig 4 (`steering_curves_three_models.pdf`).
- **Corpus**: `python3 scripts/build_steering_corpus.py` (from PD vignettes).
- **Run** (per model): `python3 scripts/run_full_pipeline.py --model E4B --stage all` (also `E2B`, `26B-A4B`).
- **Output**: vectors + eval shards on the volume under `pd_*_v1/`; aggregate with `scripts/aggregate_shards.py --run-id pd_full_v1`.

## 4. Cross-benchmark moral transfer (MoralChoice / ETHICS / MoralBench)
- **Paper**: §5.4, Table 4, App MoralBench breakdown.
- **Corpora**: `build_moralchoice_corpus.py`, `build_ethics_util_corpus.py`, `build_ethics_deontology_corpus.py`, `build_moralbench_corpus.py` (+ `build_powered_corpora.py` for n=300 arms).
- **Run**: `python3 scripts/run_external_evals.py` (and `run_powered_evals.py`, `run_moralbench_eval.py`).
- **Analyze**: `analyze_external.py` / `analyze_all_external.py` / `analyze_moralbench.py` (Wilson CIs, Fisher exact).

## 5. Reasoning-regression ablation (MMLU, GPQA-Diamond)
- **Paper**: §5.4 "No general reasoning regression".
- **Corpus**: `python3 scripts/build_reasoning_corpora.py` → `data/runs/reasoning/{mmlu,gpqa_diamond}_eval.jsonl`.
- **Run**: `python3 scripts/run_reasoning_evals.py` (cooperation vector, α ∈ {−6,−3,0,+3,+6}; GPQA at 4k-token CoT).
- **Analyze**: `aggregate_shards.py` over `shards_mmlu` / `shards_gpqa`.

## 6. Capability-regression ablation — GSM8k (math) + HumanEval (coding)  *(NEW)*
- **Paper**: §5.4 "No quantitative-reasoning or coding regression" (+ Table).
- **Corpus**: `python3 scripts/build_capability_corpora.py` → `data/runs/capability/{gsm8k,humaneval}_eval.jsonl` (GSM8k 250, HumanEval 164).
- **Run**: `python3 scripts/run_capability_evals.py` — sweeps **both** vectors (cooperation `pd_full_v1`, trust `pd_E4B_trust_v1`) at α ∈ {−6,−3,0,+3,+6}, shared α=0 baseline. Reuses `SteeringWorker.eval_shard` (no redeploy).
- **Analyze**: `python3 scripts/analyze_capability_evals.py` — pulls trace shards, scores **locally**: GSM8k numeric match, HumanEval execution-based pass@1 (sandboxed subprocess, see `game_theory_llm/capability_scoring.py`). Output: `data/runs/capability/results/aggregate.json`.
- **Tests**: `tests/test_capability_scoring.py`.
- **Result** (`data/runs/capability/results/aggregate.json`): no regression at strong steering for either vector — GSM8k cooperation 0.680/0.625/0.595 and HumanEval pass@1 0.768/0.811/0.787 across α=−6/0/+6 (all Fisher p ≥ 0.29 vs baseline). → paper §5.4 Table `tab:capability`.
- **Note**: SWE-Bench is intentionally **not** used — agentic repo-editing is infeasible for a 4B model under activation steering; HumanEval is the execution-based coding regression check.

## 7. Trust-vector experiment
- **Paper**: §5.5 (null in games), §5.4 capability ablation (row in Table).
- **Stories**: `python3 scripts/generate_trust_stories.py` (Opus; needs API key) → `build_trust_steering_corpus.py`.
- **Run**: `run_full_pipeline.py --model E4B --run-id pd_E4B_trust_v1 ...`. **Analyze**: `analyze_trust_vs_coop.py` (cosine vs cooperation vector, calibration). Runbook: `docs/trust_experiment.md`.

## 8. Multi-agent play harness + steering-in-games (ONW, Secret Hitler)
- **Paper**: §5.5, Fig 5 (`steering_in_games.pdf`), Table 5.
- **Run**: `python3 scripts/play_steering_experiment.py --game {one_night_werewolf,secret_hitler} --n-players 5 --seeds 25 --alphas="-4,0,4"` (in-container steered play on Modal).
- **Analyze**: `analyze_game_steering.py --judge claude` (Sonnet judge) + `steering_games_stats.py` (Mann–Whitney vs baseline); figure: `plot_steering_in_games.py`.
- **Report**: `docs/results/steering_in_games_report.md`.
- **Watch a match**: `python3 scripts/watch_match.py <log.jsonl> --god`.
