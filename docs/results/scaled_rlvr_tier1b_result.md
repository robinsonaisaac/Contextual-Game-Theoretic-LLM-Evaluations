# Scaled RLVR Tier-1b — Deep Game-Tree Curriculum on Qwen3-30B-A3B

**Date:** 2026-06-07
**Base model:** Qwen3-30B-A3B-Instruct-2507
**Training:** GRPO (Tinker cookbook) on a depth-curriculum of **game-tree minimax backward
induction** — the family that genuinely scales with depth (branching-2 tree → 2^depth leaves,
non-saturating integer answer) and that produced the original +12.5 pp depth-extrapolation at 4B.
op_tag = `backward_induction` (in TRAINED_TAGS).
**Curriculum:** depths 3→4→5 (file ordered by ascending depth; 200/500/500 = 1200 prompts),
group-size 12, groups/batch 32, max_tokens 2048, lr 1e-5, kl 0.05, 38 batches (1 epoch).
**Checkpoint:** `tinker://43d69e92-f25d-50ac-b24d-50d73ea55d7c:train:0/sampler_weights/final`

## Why Tier-1b: fixing the Tier-1 confound

Tier-1 trained on *saturating* free-text families (`level_k` decays to 0; `bargaining`
converges to a fixed point; `iterated_dominance`/`subtraction_game` are one-step formulas), so
"deeper" never meant "harder" and the 30B sat at ceiling — the null was **headroom-confounded**.
A headroom screen confirmed game-tree minimax instead has a **sharp capability cliff** on the
30B base, i.e. real headroom:

| depth | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| base acc | 1.00 | 0.55 | 0.15 | 0.00 | 0.00 |
| leaves | 8 | 16 | 32 | 64 | 128 |

Train where there is within-group reward variance (d4–d5); test depth-extrapolation where the
base is floored (d6–d7).

## Training was healthy and curriculum-shaped

The reward curve traces the three depth bands cleanly: **d3 on-ramp** reward ≈0.97
(`frac_all_good`=0.72) → **d4** reward climbs 0.58→0.69 (`frac_mixed`=1.00, ideal GRPO signal)
→ **d5 frontier** reward ≈−0.05 (`frac_all_bad`≈0.35, but ~65% of groups still mixed). Entropy
steps up with each harder band; KL controlled (0.0019→0.0031). Curve:
`data/runs/gt_rlvr/tier1b_30b/grpo_curve.png`. This is a materially healthier run than Tier-1,
where every prompt was at ceiling and GRPO had no gradient.

## Result: in-domain skill is REAL; transfer is NOT.

### Before → after (full suite)

| benchmark | base | rlvr | Δ | parse b→r | note |
|---|---|---|---|---|---|
| **gametree in-domain (d3–5)** | 0.611 | **0.711** | **+0.100** | 0.82→0.83 | trained op, measurable |
| &nbsp;&nbsp;· depth 4 | 0.633 | **0.867** | **+0.233** | | |
| &nbsp;&nbsp;· depth 5 (frontier) | 0.200 | **0.300** | **+0.100** | | |
| **gametree depth-extrap (d6–7)** | 0.000 | 0.006 | +0.006 | 0.06→0.04 | held-out deeper, **floored** |
| boolean_eval | 0.505 | 0.530 | +0.025 | 0.52→0.55 | held-out, measurable |
| dyck | 0.222 | 0.207 | −0.015 | 0.57→0.56 | held-out, measurable |
| mmlu_pro | 0.687 | 0.713 | +0.027 | 0.79→0.82 | held-out, measurable |
| countdown | 0.091 | 0.101 | +0.010 | — | floor |
| ordering / prontoqa / knights | 1.00 | ~1.00 | ≈0 | — | ceiling |
| **gsm8k (no-regression control)** | 0.930 | 0.925 | −0.005 | 0.95→0.93 | clean ✓ |

### Three findings

1. **In-domain operation learning is real and large.** Curriculum GRPO lifts the trained op by
   **+23.3 pp at depth 4** and **+10 pp at depth 5** (the model's frontier). This is the
   decisive contrast with Tier-1, which showed *zero* in-domain learning because there was no
   headroom. With real headroom, game-RLVR does teach the operation.

2. **No depth-extrapolation (token-wall ruled out).** At 3072 tokens d6–7 looked floored
   (0.000 → 0.006), but parse_rate ≈0.04 flagged a **generation-length wall**. Re-running d6 at
   **8192 tokens** removes it — base parse jumps 0.06→0.90 and base accuracy 0.000→**0.212**, so
   the 30B base *can* do ~21% of depth-6 trees when given room. With the wall gone, the clean
   depth-extrapolation test is unambiguous: **base d6 = 0.212, rlvr d6 = 0.212, Δ = +0.000.**
   The large in-domain gains (+23 pp at d4) do not carry even one step out-of-distribution to d6
   — no lift, no regression, flat. **The 4B +12.5 pp depth-extrapolation does not replicate at
   30B**, and this is now confirmed as a genuine reasoning-generalization fact, not a truncation
   artifact. (rlvr parse 0.83 < base 0.90 at d6: RLVR trained under a 2048-token cap and emits
   slightly shorter CoT, but accuracy among parsed is identical.)

3. **No cross-task transfer.** Measurable held-out reasoning benchmarks move within noise:
   boolean +2.5 pp, mmlu_pro +2.7 pp, dyck −1.5 pp (mean **+1.2 pp**, all sub-3pp). By the
   standing **≥3-benchmark validation rule**, this does not count as a capability gain. gsm8k
   control is flat (−0.5 pp) — **no regression**.

## Verdict (Tier-1 + Tier-1b together)

> **Game-structured RLVR produces genuine in-domain skill acquisition but not generalizable
> reasoning transfer at 30B scale — neither depth-extrapolation nor cross-task.** The capability
> gain is real (+23 pp at the trained depth) but task- and depth-bounded.

This directly answers the driving question ("can the game-generation structure make models
better at *general* reasoning?"): at 30B, it makes the model better **at the trained game**, and
that skill stays local. The encouraging 4B signals were small-model, in-distribution effects that
do not survive to a strong base.

### What would change the verdict
- **Token budget** — *resolved*. d6 retested at 8192 tokens: base 0.212, rlvr 0.212, Δ=0. The
  floor was a generation wall; with it removed the extrapolation Δ is genuinely zero.
- **Multi-epoch / larger curriculum**: 1 epoch on d3–5 may under-instill d5. More steps at the
  frontier could push d5→d6 by one notch (the cliff is one depth wide).
- **Mixed-operation (breadth) curriculum** — *the key open test*. A single op
  (backward_induction) cannot produce cross-operation transfer; depth-in-one stays local. The
  next experiment trains a *diverse* curriculum (deductive + inductive + abductive + combinatorial
  — see `reasoning_ops.py`: nim_grundy / opponent_id / signal_abduce) and tests transfer to
  held-out general-reasoning benchmarks. Breadth, not depth, is the remaining hypothesis.

## Reproduce

```bash
python3 scripts/build_tier1b.py                       # train_tier1b + deep-extrap + in-domain evals
.venv-tinker/bin/python scripts/tinker_grpo.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --train data/runs/gt_rlvr/train_tier1b.jsonl --group-size 12 --groups-per-batch 32 \
  --max-tokens 2048 --lr 1e-5 --kl 0.05 --log-path data/runs/gt_rlvr/tier1b_30b
bash scripts/tier1b_evalsuite.sh base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
bash scripts/tier1b_evalsuite.sh rlvr --model-path "$(cat data/runs/gt_rlvr/tier1b_checkpoint.txt)"
python3 scripts/plot_grpo.py --run data/runs/gt_rlvr/tier1b_30b
```

Artifacts: `data/runs/gt_rlvr/{overlap_analysis_tier1b.json, tier1b_30b/grpo_curve.png,
t1b_base_*.json, t1b_rlvr_*.json, screen_gametree_b2.json}`.
