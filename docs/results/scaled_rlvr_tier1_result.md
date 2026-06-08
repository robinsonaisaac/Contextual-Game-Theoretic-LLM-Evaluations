# Scaled RLVR Tier-1 Result — Operation-Overlap Mechanism Test (Qwen3-30B-A3B)

**Date:** 2026-06-07
**Base model:** Qwen3-30B-A3B-Instruct-2507 (non-thinking MoE, ~3B active)
**Training:** GRPO via Tinker cookbook on free-text game-theory problems
(4 families: `level_k`, `iterated_dominance`, `bargaining`, `subtraction_game`; depths 2–6;
1,200 prompts; 38 batches; reward −0.034 → ~0.10 smoothed; KL ≈ 0.0014 controlled; entropy ≈ 0.34 stable).
**Checkpoint:** `tinker://2cb36c73-b49e-5725-b828-506cf4f59d14:train:0/sampler_weights/final`
**Pre-registered trained-operation tags:** `{backward_induction, nested_belief, iterated_elimination, modular_combinatorial}`

## Verdict: **NULL at scale** — the 4B transfer effect does not survive to a strong base.

The probe-scale (Qwen3-4B) finding was a real, replicated **+12.5 pp in-domain depth-extrapolation**
gain (p≈0.02) plus a positive-but-noisy external nudge. The pre-registered Tier-1 test asked whether
that effect (a) is **operation-specific** (H_op — improvement scales with trained-operation overlap)
or (b) reflects **general output discipline** (H_gen — uniform lift). At 30B-A3B, **neither holds**:
there is no transfer, and the in-domain anchor itself fails to replicate.

### Per-benchmark (measurable cells; base accuracy in [0.10, 0.90])

| benchmark | n | base | rlvr | Δ raw | parse base→rlvr | Δ acc-among-parsed | overlap |
|---|---|---|---|---|---|---|---|
| **depth_extrap** (in-domain anchor, depths 7–8) | 160 | 0.475 | 0.431 | **−0.044** | 1.00→1.00 | **−0.044 (real)** | ov=1.0 |
| dyck | 90 | 0.489* | 0.300* | −0.189* | 0.556→0.343 | **−0.021 (≈flat)** | ov=0 |
| boolean_eval | 36 | 0.361 | 0.278 | −0.083 | 0.530→0.596 | flat (net) | ov=0 |
| mmlu_pro (knowledge contrast) | 300 | 0.693 | 0.607 | −0.087 | 0.807→0.740 | −0.040 (half drift) | ov=0 |
| gsm8k (no-regression control) | — | 0.925 | 0.930 | **+0.005** | 0.940→0.960 | clean ✓ | — |

\* `dyck` raw Δ is dominated by a parse-rate drop; accuracy-among-parsed is flat (0.418→0.397).

### Hypothesis readout
- **H_op (islands):** overlap slope **+0.020, p=0.584** (n.s.) in `dimp ~ ov + depth + C(benchmark)`.
  Improvement does **not** rise with trained-operation overlap. **Not supported.**
- **H_gen (discipline):** intercept lift +0.015, but every per-benchmark Δ is ≤ 0. **Not supported.**
- **Depth-extrapolation replication:** **FAILS at scale.** 4B: +12.5 pp (p≈0.02). 30B: −4.4 pp.
- **No-regression control:** gsm8k flat/positive (+0.5 pp), parse-rate up. GRPO did **not** damage core ability.
- **Format drift:** most held-out "regressions" are parse-rate, not reasoning, loss (acc-among-parsed
  flat on dyck/boolean; ~half of mmlu_pro). The RLVR model shifted toward the game answer format
  (`<answer>…</answer>` with a bare value), mildly hurting MC/format-sensitive parsing.

## Why null — the curriculum-ceiling confound (important caveat)

This is **not** a clean refutation of scale-transfer; it is partly a **headroom-exhaustion artifact**:

- The headroom screen at 30B-A3B showed the trained game depths (2–6) are **at ceiling**:
  `prontoqa`, `knights_knaves` = 1.00; `ordering` = 0.97; `depth_extrap` only became *measurable*
  at depths **7–8**. The 30B base already solves the curriculum the GRPO trained on.
- Consequently reward only climbed to ~0.10 (most rollouts were already correct → little group-relative
  signal). GRPO had almost nothing to teach, and the small policy shift it did induce perturbed
  output format/calibration without adding reasoning skill.

**Mechanistic reading:** the game-RLVR transfer effect is **capability-bounded** — it appears when the
base model has headroom on the trained operations (true at 4B, false at 30B for depths 2–6). At a strong
base, training on solved problems yields no reasoning gain and mild format drift. This is consistent
with — but does not by itself prove — that transfer requires *unsolved* training problems.

## Go / no-go

Per the pre-registered decision tree (H_op → Tier-2 coverage; H_gen → scale; null → run 30B point first):
we ran the 30B-A3B point and got **null**. The clean follow-up before concluding scale-independence is
**Tier-1b: re-train on deeper games (depths 6–10) where the 30B base has headroom** — i.e. match the
training curriculum to the model's frontier, then re-test depth-extrapolation (depths 11–13) and the
held-out suite. If transfer reappears with a headroom-matched curriculum, the effect is real but
curriculum-gated; if it stays null, the effect is genuinely capability-bounded to small models.

## Reproduce

```bash
# train (py3.11 Tinker venv; source main-repo .env for TINKER_API_KEY)
.venv-tinker/bin/python scripts/tinker_grpo.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --train data/runs/gt_rlvr/train_tier1.jsonl \
  --kl 0.01 --log-path data/runs/gt_rlvr/tier1_30b
# eval both arms over the 9-benchmark suite
bash scripts/tier1_evalsuite.sh base
bash scripts/tier1_evalsuite.sh rlvr --model-path "$(cat data/runs/gt_rlvr/tier1_checkpoint.txt)"
# mechanism analysis
python3 scripts/analyze_overlap.py     # -> data/runs/gt_rlvr/overlap_analysis.json
python3 scripts/plot_grpo.py --run data/runs/gt_rlvr/tier1_30b   # -> grpo_curve.png
```

Artifacts: `data/runs/gt_rlvr/{overlap_analysis.json, tier1_30b/grpo_curve.png, t1_base_*.json, t1_rlvr_*.json}`.
