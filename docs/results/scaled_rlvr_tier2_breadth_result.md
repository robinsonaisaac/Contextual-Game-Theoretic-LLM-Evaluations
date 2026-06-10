# Scaled RLVR Tier-2 — Breadth Curriculum (Deductive + Inductive) on Qwen3-30B-A3B

**Date:** 2026-06-09
**Base model:** Qwen3-30B-A3B-Instruct-2507
**Training:** GRPO (Tinker cookbook), 38 batches, group-size 12, groups/batch 32, lr 1e-5, kl 0.05,
**max_tokens 6144** (token-wall-safe — the lesson from Tier-1b).
**Curriculum:** difficulty tiers mixing two operations with real 30B headroom —
**game-tree minimax** (deductive, `backward_induction`) + **opponent-ID** (inductive,
`inductive_rule`): tier1 {gt d3, opp d1} → tier2 {gt d4, opp d2} → tier3 {gt d5, opp d3},
200 prompts per (op, depth) = 1200 total, balanced 600/600.
**Checkpoint:** `tinker://a7b60660-1559-5d1a-a697-2ba8088b0b97:train:0/sampler_weights/final`
(run paused once on a Tinker billing block at batch 16; resumed cleanly from the batch-10 checkpoint).

## Why Tier-2: the breadth hypothesis

Tier-1b established that **depth-in-one-operation** produces real in-domain skill (+23 pp at the
trained depth) but zero transfer — no depth-extrapolation (clean Δ=0 at 8192 eval tokens) and no
cross-task movement. The remaining hypothesis was **breadth**: a curriculum *diverse in reasoning
operations* (deductive + inductive — the only two of five candidate ops with 30B headroom; closed-form
nim/abduction sit at ceiling) might produce the generalizable gains a single op could not.

## Training was the healthiest of the three runs

Reward traces the curriculum tiers (0.80 → 0.50 → 0.35 as tiers harden — band difficulty, not
regression). Crucially, at the hardest tier **frac_mixed ≈ 0.94 with frac_all_bad ≤ 0.19** —
compare Tier-1b's frontier tier where 31–44 % of groups were all-bad (truncation-starved at 2048
training tokens). The 6144-token budget kept GRPO gradient alive everywhere. Entropy rose through
tiers 1–2 (exploration) then dropped sharply at tier 3 (policy converging). KL controlled (~0.0022).
Curve: `data/runs/gt_rlvr/tier2_30b/grpo_curve.png`.

## Result: breadth does NOT unlock transfer — and dilutes in-domain learning.

### Before → after (full suite; eval budgets identical across arms)

| benchmark | base | rlvr | Δ | note |
|---|---|---|---|---|
| in-domain (d3–5 / d1–3) | 0.544 | 0.572 | **+0.028** | weak vs Tier-1b's +0.100 |
| extrapolation (gt d6 + opp d4) | 0.100 | 0.113 | +0.012 | flat |
| boolean_eval | 0.510 | 0.515 | +0.005 | held-out, measurable |
| dyck | 0.242 | 0.227 | −0.015 | held-out, measurable |
| mmlu_pro | 0.677 | 0.677 | +0.000 | held-out, measurable |
| bbh_hard | 0.830 | 0.843 | +0.013 | held-out, measurable |
| **gsm8k (control)** | 0.920 | 0.920 | **+0.000** | clean ✓ |

**Mean held-out transfer = +0.001** (4 measurable benchmarks) vs Tier-1b depth-only **+0.012** —
both are noise. The ≥3-benchmark rule fails decisively.

### Per-operation breakdown

| op | depth | base | rlvr | Δ |
|---|---|---|---|---|
| gametree | d3 | 1.000 | 1.000 | 0 (ceiling) |
| gametree | **d4** | 0.767 | **0.900** | **+0.133** |
| gametree | d5 | 0.433 | 0.433 | 0 |
| gametree | d6 (extrap) | 0.150 | 0.150 | 0 |
| opponent_id | d1 | 0.567 | 0.567 | 0 |
| opponent_id | d2 | 0.300 | 0.333 | +0.033 |
| opponent_id | d3 | 0.200 | 0.200 | 0 |
| opponent_id | d4 (extrap) | 0.050 | 0.075 | +0.025 |

Two findings inside the null:

1. **The breadth–depth tradeoff is real and roughly proportional.** At fixed budget (1200 prompts),
   halving per-cell data halved the in-domain gain: gametree d4 got **+13.3 pp** here with 200
   prompts vs **+23.3 pp** in Tier-1b with 500. Spreading the budget across operations bought no
   transfer and cost in-domain learning.
2. **The inductive op barely trained.** opponent_id moved ≤ +3.3 pp everywhere despite live reward
   signal (tier-3 reward ~0.35, mixed ≈ 0.94). Inferring a 2^k rule table appears RLVR-resistant at
   this scale/budget — group-relative advantages reward lucky guesses on a small answer space as
   much as genuine table inference.

## Combined verdict (Tier-1 + Tier-1b + Tier-2)

Three pre-registered attempts to convert game-structured RLVR into general reasoning gains on a
strong 30B base:

| attempt | design | in-domain | transfer |
|---|---|---|---|
| Tier-1 | 4 saturating ops, d2–6 | none (ceiling, no headroom) | null |
| Tier-1b | 1 scaling op, deep curriculum, 2048 tok | **+23 pp** at frontier | null (clean Δ=0) |
| Tier-2 | 2 ops (deductive+inductive), tiered, 6144 tok | +13 pp (diluted) | **null (+0.001)** |

> **Game-structured RLVR on a strong base produces only local, trained-cell skill — neither
> depth, nor adequate token budgets, nor operation diversity converts it into transferable
> reasoning.** The 4B-scale transfer signals were small-model, in-distribution effects.

What survived every test: the **no-regression guarantee** (gsm8k exactly flat in all three runs)
and the methodology itself (headroom screening, token-wall checks, pre-registered op-overlap,
≥3-benchmark rule) — which is what made these nulls clean rather than ambiguous.

### If anyone pushes further (not currently planned)
- **Scale data/epochs at the frontier cell** (gt d4-d5, thousands of prompts, multi-epoch) — tests
  whether transfer has a data threshold; Tier-1b/2's proportionality suggests in-domain keeps
  growing, but nothing yet hints transfer follows.
- **Process rewards** (per-step verification from the game solver, not outcome-only) — the game
  structure gives exact intermediate states free; outcome-only GRPO may be the limiting factor.
- **Weaker base (4B/8B) breadth replication** — where every op has headroom, to test whether
  breadth-transfer exists at all anywhere.

## Reproduce

```bash
python3 scripts/build_tier2_breadth.py
.venv-tinker/bin/python scripts/tinker_grpo.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --train data/runs/gt_rlvr/train_tier2_breadth.jsonl --group-size 12 --groups-per-batch 32 \
  --max-tokens 6144 --lr 1e-5 --kl 0.05 --log-path data/runs/gt_rlvr/tier2_30b
bash scripts/tier2_evalsuite.sh base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
bash scripts/tier2_evalsuite.sh rlvr --model-path "$(cat data/runs/gt_rlvr/tier2_checkpoint.txt)"
python3 scripts/analyze_tier2.py     # -> data/runs/gt_rlvr/tier2_breadth_analysis.json
```

Artifacts: `data/runs/gt_rlvr/{tier2_breadth_analysis.json, tier2_30b/grpo_curve.png,
t2_base_*.json, t2_rlvr_*.json}`.
