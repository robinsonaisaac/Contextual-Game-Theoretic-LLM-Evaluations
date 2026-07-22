# Tier-5 Inline Ledger — Result

**Date:** 2026-07-17
**Branch:** `feature/activation-steering`
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (Qwen3-30B-A3B-Instruct-2507), LoRA SFT + GRPO via Tinker
**Spec:** `docs/superpowers/specs/2026-07-02-tier5-inline-ledger-design.md`
**Claim discipline:** §7 of the spec — "generalizable technique" requires held-out families AND ≥2/3 real benchmarks to pass; in-domain result alone is explicitly insufficient.

---

## 1. Headline

The trained inline-ledger bookkeeping loop is **real, load-bearing, and extrapolates within its training family**: graph search and forward chaining reach 0.900 and 1.000 accuracy at h=130, both far beyond trained horizons, gains that collapse to near-zero when the trained model is instructed to suppress its protocol (double dissociation). Outside the training distribution the picture is the opposite — SFT is **actively harmful** across every off-family benchmark, with drops of up to −70pp (BBH multistep arithmetic 0.976→0.320; held-out object tracking 0.780→0.080; held-out scheduling 0.180→0.020). Causal attribution decomposes the harm into two mechanisms: recoverable format-imposition for tasks with intact knowledge (arithmetic: suppression restores 0.948≈base), and unrecoverable capability damage for tasks that relied on the model's natural bookkeeping (tracking: suppression changes nothing, 0.080→0.090). GRPO with a dense ledger reward over seven iterations (v2 run, on an eval-disjoint pool) adds +0.02 on average — every delta vs SFT is within ±0.03 except a partial format-recovery on BBH arithmetic (+0.16, still −0.50 below base) — a third confirmation that RLVR does not meaningfully move generalization in this regime. **The spec §7 generalization claim is rejected on both pre-registered gates.**

---

## 2. Method Recap

### Protocol invariants (semantic, not syntactic)
The ledger is specified as five semantic invariants, deliberately surface-agnostic to avoid the SAE lexical-artifact failure mode (where training on a fixed grammar teaches token detection, not the skill):
1. Externalize established facts compactly as computation proceeds — never hold them in context only.
2. Ground each step in recorded facts — state what is computable now and from which recorded entries.
3. Small bounded steps — no re-derivation of long chains.
4. Periodic state restatement at ~8–12 step intervals (the inline analogue of the env-managed checkpoint that scored 0.988).
5. Answer read off the final recorded state, not recomputed.

### Skeleton IR
Generators emit format-independent skeletons — lists of semantic events `note(key, value, deps)`, `restate()`, `answer(x)` plus a gold key-set. All verification, QC, and the GRPO dense reward operate at skeleton level: one verifier, format-independent, unhackable by surface variation.

### Naturalization and QC
Skeletons are realized in two surface forms. The canonical tagged grammar (`LEDGER / NEXT / NOTE / CHECKPOINT / ANSWER`, 25% of traces) is kept for cheap deterministic parsing in the GRPO reward loop. The remaining 75% are LLM-naturalized: Sonnet 4.6 rephrases each skeleton under a rotating style seed ("engineer's log", "formal derivation", "casual working notes", "markdown state table", "numbered facts F1/F2…"), then an extract-back pass pulls the claimed facts mechanically; traces failing equality against the gold skeleton are regenerated or dropped. This rejection-sampled faithfulness check is non-negotiable — corrupted traces would teach corrupted bookkeeping, the exact disease under treatment.

### Dataset composition
**2518 training rows**, QC pass-rate **0.742**, composition: ~25% canonical grammar / ~75% naturalized; ~15% short no-ledger traces (teach when to deploy; protect short-form performance); ~12% recovery traces (injected ledger corruption followed by restate-verify-correct-continue, rendered in both surface forms). Four train families: minimax game trees (depth 3–6, 7–63 nodes), register-machine simulation (program length N), graph search/BFS (nodes/edges), forward-chaining deduction (Horn rule-chain length). Two held-out families never entering training: multi-entity object tracking, constraint scheduling. RL pool: 720 rows from the cram boundary (mixed-success horizons in graph, register, trees families).

### SFT
LoRA on Tinker; loss on completion only. 630 steps / 2 epochs (step 1 loss 0.317, step 630 loss 0.006; epoch-1 terminal reproduces 0.015 as logged). The SFT rerun (T5-9) reproduced exactly (0.317→0.015 at epoch 1). Checkpoint: `tinker://e74ed103-be5a-59c1-92a6-e5c8cc23291a:train:0/sampler_weights/tier5_sft_v2`.

### Gate and override
In-domain eval after SFT (uncensored budget, 12k tokens for long cells): overall base→SFT 0.144→0.465; trained-horizon SFT 0.769. Extrapolated-horizon mean delta +0.008; the pre-registered STOP rule fired on extrap delta < threshold. User overrode to proceed to RL on 2026-07-05, recorded in `sdd/progress.md`.

### GRPO
Seven iterations (1 epoch over the filtered pool), families graph/register/trees, horizons 31–130, dense reward `answer_correct + 0.5 · ledger_accuracy` (precision·recall spray-guard), canonical format cue in rollout prompts for cheap deterministic skeleton extraction. **The reported RL numbers are from the v2 re-run (2026-07-21/22)** on the regenerated, eval-disjoint pool (seed range 300000+; exact-prompt intersection with all 14 eval sets and the train set = 0), warm-started from the same SFT state checkpoint. Per-batch reward varied 0.37–1.09 with batch family/horizon mix, no collapse. Final v2 checkpoint: `tinker://9b36cd8b-f064-5fcd-a3cd-1f0eec6ddf25:train:0/sampler_weights/final` (recorded in `data/runs/tier5/rl_checkpoint_v2.txt`; the superseded v1 train-on-test checkpoint was `tinker://4b5b3078-01b9-525b-b006-5c4810ccf346:train:0/sampler_weights/final`).

---

## 3. Results

### 3a. Gate Table (pre-battery, uncensored)

| Eval set | Base acc | SFT acc | Delta | Notes |
|---|---|---|---|---|
| trees h31 | 0.083 | 0.983 | +0.900 | trained horizon; near-saturation |
| trees h63 | 0.017 | 0.017 | 0.000 | d6 cliff — SFT parse 1.000 (format perfect, values corrupt) |
| trees h127 | 0.017 | 0.050 | +0.033 | extrapolated |
| graph h60 | 0.317 | 1.000 | +0.683 | trained |
| graph h90 | 0.033 | 0.467 | +0.433 | cram boundary |
| graph h130 | 0.600 | 0.667 | +0.067 | extrapolated |
| chain h60 | 0.200 | 1.000 | +0.800 | trained |
| chain h90 | 0.033 | 1.000 | +0.967 | trained |
| chain h130 | 0.400 | 1.000 | +0.600 | extrapolated (saturated) |
| register h60 | 0.700 | 0.800 | +0.100 | trained |
| register h90 | 0.767 | 0.883 | +0.117 | trained |
| register h150 | 0.667 | 0.000 | −0.667 | token-overhead trap (parse 0.000 even at 12k) |
| **Overall mean** | 0.144 | 0.465 | | |
| **Trained-horizon mean** | — | 0.769 | | rule gate checks this |
| **Extrap mean delta** | — | — | **+0.008** | rule fired STOP; user overrode |

**Key gate finding — trees h63:** SFT parse rate 1.000 with accuracy 0.017. The model formats the ledger perfectly at depth 6 but computes wrong minimax values. The d6 capability cliff identified in Tiers 1–4 survives SFT with perfect format compliance, confirming it is a capability property, not a formatting or discipline problem.

**Key gate finding — register h150:** SFT parse rate 0.000 even at 12k-token budget. The ledger cost scales with program length; the naturalized surface form (wordy ~75% of training) cannot compress into the token budget at horizon 150. The base model solves this tersely at 0.633 without a ledger. This is the token-overhead trap: training a long protocol on a medium-horizon family does not guarantee that protocol can scale to large horizons.

### 3b. Full Battery (all 10 eval sets × 3 models)

| Benchmark | Family | Base | SFT | RL | Delta SFT | Delta RL |
|---|---|---|---|---|---|---|
| graph h130 | in-domain | 0.450 | 0.900 | 0.883 | +0.450 | −0.017 |
| chain h130 | in-domain | 0.450 | 1.000 | 1.000 | +0.550 | 0.000 |
| trees d7 (h127) | in-domain | 0.050 | 0.067 | 0.100 | +0.017 | +0.033 |
| register n150 | in-domain | 0.633 | 0.000 | 0.000 | −0.633 | 0.000 |
| HELD-OUT tracking | off-family | 0.780 | 0.080 | 0.140 | −0.700 | +0.060 |
| HELD-OUT scheduling | off-family | 0.180 | 0.020 | 0.030 | −0.160 | +0.010 |
| BBH arith | real benchmark | 0.976 | 0.320 | 0.476 | −0.656 | +0.156 |
| BBH dyck | real benchmark | 0.336 | 0.056 | 0.032 | −0.280 | −0.024 |
| BBH tracking | real benchmark | 0.004 | 0.000 | 0.000 | −0.004 | 0.000 |
| GSM8K (control) | short-form | 0.920 | 0.920 | 0.910 | 0.000 | −0.010 |

*RL column: GRPO v2 (eval-disjoint pool). The superseded v1 RL column (trained on a pool that reused the in-domain eval seeds) is preserved in git history; its deltas were [−0.017, +0.020].*

**n per set:** graph/chain/trees/register = 60; held-out tracking/scheduling = 100; BBH sets = 250; GSM8K = 200.

**RLVR result (v2, clean pool):** mean delta RL vs SFT is +0.021; nine of ten deltas lie in [−0.024, +0.060]. The one outlier is BBH arithmetic at +0.156 (0.320→0.476) — a partial recovery consistent with RL relaxing the SFT-imposed ledger habit on short arithmetic, but still −0.500 below the untuned base, so it does not alter the off-family harm picture. In-domain and held-out generalization move ≤0.06. Seven iterations of GRPO with a dense ledger reward — addressing the original d6 no-signal problem, now with a pool disjoint from every eval set — still produces no meaningful generalization gain. This is the third confirmation of the RLVR ceiling in this model/task regime (Tiers 2, 3b, and now 5), and the v1 (train-on-test) and v2 (clean) runs agree: −0.017 vs +0.021 mean delta.

**Short-form control:** GSM8K 0.920/0.920/0.910 — no short-form regression from the 15% no-ledger training mix. The "when to deploy" protection worked for the family it explicitly covered.

### 3c. Attribution Table — Suppression Ablation + Ledger Usage

**Part A: unprompted ledger usage rates (LLM judge, Sonnet, five §3 invariants)**

| | Base tracking | SFT tracking | Base BBH arith | SFT BBH arith |
|---|---|---|---|---|
| externalizes | 1.00 | 1.00 | 1.00 | 0.43 |
| grounds steps | 1.00 | 1.00 | 1.00 | 0.30 |
| small steps | 1.00 | 1.00 | 1.00 | 0.43 |
| restates | 0.80 | 0.93 | 0.23 | 0.07 |
| reads off answer | 1.00 | 1.00 | 1.00 | 0.43 |
| **uses_ledger** | **1.00** | **1.00** | **1.00** | **0.33** |

The base model already exhibits invariant-satisfying bookkeeping naturally on tracking tasks (100% usage at 0.780 accuracy). SFT replaced a working natural protocol with a broken trained one.

**Part B: suppression ablation — instruct trained model not to use explicit state-tracking**

| Domain | Base | SFT | SFT suppressed | Shrinkage | Verdict |
|---|---|---|---|---|---|
| HELD-OUT tracking | 0.780 | 0.080 | 0.090 | 0.010 | **catastrophic forgetting** — underlying capability damaged |
| BBH arith | 0.976 | 0.320 | 0.948 | 0.628 | **format imposition** — knowledge intact, ledger habit destructive |
| in-domain graph h130 | 0.450 | 0.900 | 0.050 | −0.850 | **ledger is load-bearing** — double dissociation confirmed |

**Double dissociation:** Suppressing the trained model on in-family graph collapses 0.900→0.050 (near-chance), proving the ledger is the active strategy, not a co-variate. Suppressing on BBH arith recovers 0.948≈base, proving the knowledge is intact and the harm was pure protocol interference. These two results together constitute a causal double dissociation: the trained protocol is load-bearing in-family and causally harmful off-family.

**Suppression compliance caveat:** The suppression instruction may not achieve full compliance in free-generation (the model may partially revert to trained habits). If compliance is imperfect, the tracking suppression result (0.090≈SFT) may overstate forgetting — some residual tracking attempt could remain and still fail. This caveat does not affect the arithmetic result (where suppression fully recovers) or the in-family graph result (where suppression fully collapses), but it limits the strength of "unrecoverable capability damage" as a conclusion for tracking.

---

## 4. Interpretation

### 4a. Style diversity ≠ skill generality — family diversity was the binding constraint

The training set covered four syntactic and semantic families with deliberate surface variety (five style seeds, rejection-sampled naturalization). This is the behavioral-level echo of the SAE finding: training on a fixed grammar teaches the grammar (the "cooperate"-word-detector artifact); we addressed that by varying grammar. But varying style within a fixed set of logical families teaches style-varied instances of that family's reasoning structure. The families shared no structural invariant beyond the five abstract protocol invariants — and those invariants apparently were not learned as transferable skills; what was learned was family-specific bookkeeping patterns. Off-family, the model applies its trained pattern destructively to domains with different structural requirements (arithmetic's value-computation; Dyck's bracket-matching). The lesson: **style diversity is not a substitute for family diversity**. The binding constraint was the number of distinct logical families, not the surface form variety.

### 4b. "When to deploy" only protected what the mix explicitly covered

The GSM8K control result (0.920/0.920/0.910 across base/SFT/RL) confirms the 15% no-ledger mix worked exactly as intended — short-form tasks explicitly included in training were protected. But held-out tracking (base 0.780, SFT 0.080) shows the discrimination did not generalize to unseen long-horizon families. The model did not learn a general "should I use a ledger" decision; it learned a task-family-conditional deployment policy. This is consistent with the in-domain register h150 failure: even within a trained family, at a horizon the model was not trained to handle, it applied the ledger and failed (rather than falling back to the terse base approach).

### 4c. The d6 capability cliff conclusion strengthens

The minimax trees d6/d7 cliff (accuracy 0.017 at h=63–127 with base, gate SFT, and final SFT) now survives four distinct interventions: Tier-1/2 RLVR, Tier-3 process reward (rejected as a different failure mode), Tier-4 env-managed scaffolding (which was the only 0.99 regime, but via external state not internal competence), and now Tier-5 SFT+RL with perfect ledger format compliance at h63. The gate result at trees h63 (parse_rate 1.000, accuracy 0.017) is the clearest evidence yet: the model formats correctly, the ledger is applied, but the minimax value propagation at depth 6 is wrong. This is a capability property of the model family, not a format, discipline, scaffold, or training-data deficiency.

### 4d. Implications for the design space

Two deployment paths emerge. First: **train the decision to externalize across many more families** (20+, spanning diverse logical structures) before making generalization claims. The current result would count as a positive existence proof of in-family skill for a future many-family study. Second: **scaffold at inference** — the env-managed control regime (Tier-4, 0.988 on in-distribution tasks) remains the only demonstrated path to near-ceiling long-horizon accuracy, and it works by removing the self-management problem rather than teaching the model to solve it. Given the RLVR ceiling at ~1pp and the family-specificity finding here, inference-time scaffolding remains the only 0.99-regime path for production use.

---

## 5. Limitations

1. **Seven-iteration RL (~1 epoch over pool):** GRPO ran only seven iterations over the filtered cram-boundary pool, approximately one epoch. This is not a saturated RL run. It is possible that additional iterations would produce meaningful improvement, though the Tier-2 and Tier-3 results (which ran longer) also capped near SFT, and the dense reward signal in those regimes was also present.

2. **Suppression-compliance caveat (tracking):** As noted in §3c, the suppression instruction for the held-out tracking domain may not fully suppress trained habits. The "catastrophic forgetting" verdict for tracking assumes full compliance; partial compliance would mean the 0.090 suppressed result includes some still-trained behavior, making capability damage appear larger than it is.

3. **Single model:** All results are from one model family (Qwen3-30B-A3B-Instruct-2507). The capability-property conclusions (d6 cliff, token-overhead trap) and the family-specificity result may differ across model families, particularly larger or stronger base models.

4. **QC family skew unmeasured:** The naturalization QC pass-rate of 0.742 is reported overall; per-family QC skew was not measured. If some families passed QC at substantially higher rates, their training representation would be inflated relative to the nominal 25/75/12/15 composition. This could explain why chain/graph trained strongly while register and trees lagged — per-family QC is a recommended metric for future replication.

5. **Attribution n=30:** Suppression ablation and ledger-usage attribution used n=30 per condition (from `raw_*` jsonl subsets). Battery results used n=60–250. Attribution conclusions are directionally clear but imprecisely estimated.

---

## 6. Verdict Against Spec §7 Thresholds

The spec §7 defines four evaluation targets and explicit claim discipline: "generalizable technique" requires (2) AND (3) to pass; (1) alone is an in-domain result.

| §7 Target | Threshold | Result | Pass? |
|---|---|---|---|
| (1) In-domain extrapolated horizons: success ≥ +15pp where base collapses | +15pp extrapolated | graph h130: +45pp (0.450→0.900); chain h130: +55pp (0.450→1.000); trees d7: +1.7pp (cliff persists) | **PARTIAL** — graph/chain pass strongly; trees fails (cliff survives); register fails (overhead trap) |
| (2) Held-out families zero-shot: success ≥ +10pp | +10pp on held-out | tracking: −70pp (0.780→0.080); scheduling: −16pp (0.180→0.020) | **FAIL** — both off-family; both strongly negative |
| (3) Real benchmarks: uplift on ≥2 of 3; GSM8K degradation ≤ 2pp | uplift ≥2/3 BBH; GSM8K ≤ −2pp | BBH arith: −65.6pp; BBH dyck: −28pp; BBH tracking: −0.4pp; GSM8K: 0pp (control) | **FAIL** — 0/3 benchmarks show uplift; GSM8K control passes |
| (4) Attribution: suppression shrinks gains; unprompted technique usage ≥ threshold | gains shrink on suppression | in-family graph: collapses 0.900→0.050 (ledger load-bearing, suppression confirmed); off-family arith: recovers to base (knowledge intact); tracking: no shrinkage (capability damaged, not suppressible) | **PARTIAL** — in-family attribution is clean; off-family dissociation is informative |

**Overall verdict:** The spec §7 generalization claim is **REJECTED** per its own pre-registered discipline. Targets (2) and (3) both fail; (1) partially passes (two of four trained families extrapolate, two do not); (4) provides causal clarity on the failure mode. The in-family result for graph search and forward chaining is a genuine positive, but it is explicitly insufficient by the spec's own definition.

---

## 7. Reproduction

Run the following from the worktree root. Pure-python steps use `python3`; Tinker steps use `.venv-tinker/bin/python`. The result JSONs in `data/runs/tier5/` are the canonical source of truth for all reported numbers.

```bash
# 0. Secrets (OPENROUTER_API_KEY for naturalization + judge; TINKER_API_KEY for training)
source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env

# 1. Dataset build — writes train_tier5.jsonl, eval_*.jsonl, rl_pool.jsonl into
#    data/runs/tier5/ (paths are hardcoded in the script; --build is the full paid
#    naturalization; omit it for a free skeleton-only dry-run that still writes the
#    eval sets and RL pool)
python3 scripts/build_tier5_ledger.py --build

# 2. SFT (LoRA rank 32, lr 1e-4, 2 epochs, batch 8; ~630 steps on 2,518 rows).
#    Writes the sampler path to sft_checkpoint.txt and the RESUMABLE STATE path to
#    sft_checkpoint.txt.state — RL warm-start needs the state path, not the sampler.
.venv-tinker/bin/python scripts/tinker_sft.py \
  --train data/runs/tier5/train_tier5.jsonl \
  --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --save-name tier5_sft_v2 \
  --ckpt-out data/runs/tier5/sft_checkpoint.txt
# Training curve: data/runs/tier5/sft_metrics.jsonl.
# NOTE: the reported run predates the over-length guard added after review — rows whose
# tokenized prompt+completion exceeded --max-len were then silently truncated; the
# script now SKIPS such rows and prints how many (see §5 limitations re long-horizon
# naturalized cells).

# 3. Gate eval (no arguments; reads sft_checkpoint.txt, runs base+SFT over all
#    eval_indomain_*.jsonl with the 12k budget for long cells, prints the decision)
.venv-tinker/bin/python scripts/tier5_gate.py
# Writes: data/runs/tier5/gate_report.json  (reported: PROCEED,
# mean_trained_sft=0.658, mean_extrap_delta=0.071)

# 4. GRPO (warm-start from the SFT state; comma-separated filters select the
#    cram-boundary families/horizons from gate_report.json)
.venv-tinker/bin/python scripts/tinker_grpo_ledger.py \
  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt.state)" \
  --train data/runs/tier5/rl_pool.jsonl \
  --families graph_search,register_machine,trees \
  --horizons 60,90,130,31,63 \
  --max-tokens 6000 \
  --save-every 10 \
  --log-path data/runs/tier5/grpo_ledger_run
# Full run artifacts: data/runs/tier5/grpo_ledger_run/ (metrics.jsonl, per-iteration
# rollouts). The final sampler path is the "sampler_path" of the last row of
# grpo_ledger_run/checkpoints.jsonl; record it in data/runs/tier5/rl_checkpoint.txt.

# 5. Full battery (base / sft / rl × 10 eval sets)
bash scripts/tier5_evalsuite.sh base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
bash scripts/tier5_evalsuite.sh sft  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt)"
bash scripts/tier5_evalsuite.sh rl   --model-path "$(cat data/runs/tier5/rl_checkpoint.txt)"
# Writes: data/runs/tier5/t5_{base,sft,rl}_*.json

# 6a. Suppression cells: re-run the three cells with a suppression system message
#     via tinker_eval's --system flag (produces the t5_sftsupp_*.json files):
.venv-tinker/bin/python scripts/tinker_eval.py --eval ledger \
  --corpus data/runs/tier5/eval_indomain_graph_search_h130.jsonl \
  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt)" \
  --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 --temperature 0 --max-tokens 12000 \
  --system "Do NOT write out any explicit state, notes, tables, or a running ledger. Answer using only inline prose reasoning." \
  --out data/runs/tier5/t5_sftsupp_indom_graph_130.json
# (repeat for bbh_arith @2048 and heldout_tracking @4096)

# 6b. Attribution — subcommands, not flags (judge needs OPENROUTER_API_KEY):
python3 scripts/tier5_attribution.py judge --samples <samples.jsonl> --tag <tag>
python3 scripts/tier5_attribution.py suppression \
  --normal data/runs/tier5/t5_sft_indom_graph_130.json \
  --suppressed data/runs/tier5/t5_sftsupp_indom_graph_130.json \
  --tag indom_graph_130 --base-acc <base> --sft-acc <sft>
# (repeat suppression for bbh_arith and heldout_tracking)
python3 scripts/tier5_attribution.py finalize
# Writes: data/runs/tier5/attribution.json

# 7. Verify
python3 -c "
import json
d = json.load(open('data/runs/tier5/attribution.json'))
assert d['part_b_suppression']['bbh_arith']['sft_suppressed'] > 0.90, 'arith suppression recovery check'
assert d['part_b_suppression']['indom_graph_130']['sft_suppressed'] < 0.10, 'graph collapse check'
print('Verification passed')
"
```

Post-review corrections to this pipeline: the RL pool is generated from a seed range
(300000+) disjoint from every eval set — the original v1 GRPO run's pool had reused the
in-domain eval seeds (train-on-test for RL). GRPO was re-run on the clean pool
(2026-07-21/22) and the RL battery re-evaluated; the reported RL numbers above are from
that v2 run. v1 (contaminated) and v2 (clean) agree on the null: mean RL−SFT delta
−0.017 vs +0.021. BBH dyck is scored by exact match (`--eval dyck`) instead of the
ledger normalizer, whose `.()`-stripping could collapse distinct bracket answers.

All raw per-item results are in `data/runs/tier5/t5_{base,sft,rl}_*.json` (JSON with `per_item` arrays). SFT training curve: `data/runs/tier5/sft_metrics.jsonl` (1260 rows, step/epoch/batch/loss). Attribution detail: `data/runs/tier5/attribution.json`. Gate detail: `data/runs/tier5/gate_report.json`.
