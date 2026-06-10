# Tier-3 — Process Reward vs Outcome Reward, Head-to-Head (Qwen3-30B-A3B)

**Date:** 2026-06-10
**Design (pre-registered):** two GRPO arms identical in every respect — same base model, same
1200 problems (the exact Tier-1b game-tree set, d3:200/d4:500/d5:500 curriculum), same 6144
training tokens, same hyperparameters (group 12, gpb 32, lr 1e-5, kl 0.05, 38 batches) —
differing **only in the reward**:
- **OUTCOME** (control): reward = 1{final answer correct}
- **PROCESS** (treatment): reward = 0.5·outcome + 0.5·process, where process = order-aware
  LCS recall of the solver's gold internal-node value sequence (exact step labels, free by
  construction; spray-guarded — see `process_reward.py`, 9 unit tests).

**Hypothesis tested:** outcome-only reward teaches depth-specific shortcuts (Tier-1b/2:
in-domain gains, zero extrapolation); rewarding the *procedure* should generalize one depth
deeper. **Primary endpoint:** d6 extrapolation (never-trained depth, n=160, 8192 eval tokens).
**Checkpoints:** outcome `tinker://5f079616…final`, process `tinker://285785e5…final`.

## Verdict: process hypothesis REJECTED — and the outcome arm's apparent extrapolation
## downgraded to a non-significant trend under deterministic decoding.

### Primary endpoint: d6 extrapolation (greedy, temp=0 — deterministic)

| arm | d6 acc | parse | vs base | vs outcome |
|---|---|---|---|---|
| base | 0.144 | 0.800 | — | — |
| **outcome** | **0.206** | 0.787 | +6.2 pp (z=+1.47, **p=0.141**) | — |
| process | 0.181 | 0.806 | +3.8 pp (z=+0.91, p=0.363) | **−2.5 pp (p=0.572)** |

Process does not beat outcome (wrong direction under both decoding regimes); neither arm
significantly beats base at n=160.

### Secondary (temp-0.7 suite)

| metric | base | outcome | process |
|---|---|---|---|
| d6 (temp 0.7) | 0.113 | 0.263 | 0.212 (vs outcome: −5.0 pp, p=0.293) |
| in-domain d4 | 0.775 | **0.900** | 0.850 |
| in-domain d5 | 0.550 | 0.550 | **0.325** |
| transfer mean (boolean/dyck/mmlu/bbh) | — | +0.003 | +0.003 |
| gsm8k control | 0.920 | 0.930 | 0.930 |

1. **Process is *worse* in-domain** (d5: 0.325 vs 0.550) — splitting reward weight between
   procedure narration and answer correctness diluted answer pressure; the model that narrates
   values is not the model that gets deep answers right.
2. **Transfer is identically null for both arms** (+0.003 each) — reward type does not move
   cross-task transfer at all. gsm8k clean in both.
3. **The dead-group prediction was mooted by the token fix**: at 6144 training tokens, BOTH
   arms had frac_all_bad = 0.00 at the hardest tier (Tier-1b's 31–44 % dead groups at 2048
   tokens were starvation, not a reward problem). Dense process credit had nothing left to fix.

### The outcome-extrapolation question (the run's most interesting thread)

The temp-0.7 suite showed outcome d6 = 0.263 vs base 0.113 (+15 pp) — which would have been
the first positive extrapolation at 30B and implied *training-token budget* unlocks depth
generalization (Tier-1b's 2048-token outcome arm had a clean Δ=0). The greedy re-run shrinks
this to **+6.2 pp, p=0.141** — a consistent positive trend (outcome > process > base in both
regimes) but not significant at n=160. Honest read: 6144-token outcome training *may* buy a
small extrapolation gain; confirming it needs ~n≥500 per arm or multi-seed evals. It is firmly
smaller than the in-domain gain (+12.5 pp at d4), so the local-skill conclusion stands.

### Methodological finding: temp-0.7 single-run eval stochasticity

Base d6 measured **0.212 / 0.113 / 0.144** across three runs (two at temp 0.7 on overlapping
items, one greedy). Single temp-0.7 runs carry material run-to-run variance on hard benchmarks
— enough to manufacture or erase a ±10 pp "effect." All small-effect endpoints in this project
should use greedy decoding or multi-seed means. (Prior tiers' conclusions are robust: they
rested on +23 pp / Δ≈0 / 4-benchmark-mean contrasts, not single ±5 pp cells.)

## Where this leaves the research program

Four pre-registered designs (Tier-1 scaled ops, Tier-1b deep curriculum, Tier-2 breadth,
Tier-3 process rewards) all converge: **game-structured RLVR on a strong 30B base reliably
produces in-domain skill (up to +23 pp at trained depths, never a regression elsewhere) but no
validated transfer — not across depth, not across operations, not across tasks, regardless of
reward signal.** The remaining untested lever from the Tier-2 doc is data/epoch scaling at the
frontier cell, which all evidence to date suggests buys more in-domain skill only.

## Reproduce

```bash
python3 -m pytest tests/test_process_reward.py -q          # reward unit tests
python3 scripts/build_tier3_process.py                     # train + eval corpora
.venv-tinker/bin/python scripts/tinker_grpo.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --train data/runs/gt_rlvr/train_tier3.jsonl --reward outcome --group-size 12 \
  --groups-per-batch 32 --max-tokens 6144 --lr 1e-5 --kl 0.05 --log-path data/runs/gt_rlvr/tier3_outcome_30b
# (same with --reward process --log-path .../tier3_process_30b)
bash scripts/tier3_evalsuite.sh outcome --model-path "$(cat data/runs/gt_rlvr/tier3_outcome_checkpoint.txt)"
bash scripts/tier3_evalsuite.sh process --model-path "$(cat data/runs/gt_rlvr/tier3_process_checkpoint.txt)"
# greedy primary endpoint: tinker_eval --temperature 0 on eval_t3_d6.jsonl for all 3 arms
python3 scripts/analyze_tier3.py
```

Artifacts: `data/runs/gt_rlvr/{tier3_analysis.json, t3_*_*.json, t3g_*_d6.json,
tier3_{outcome,process}_30b/grpo_curve.png}`.
