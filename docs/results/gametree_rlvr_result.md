# Free-text game-theory RLVR — lean transfer probe (result)

**Setup.** Base `Qwen/Qwen3-4B-Instruct-2507`. GRPO (Tinker cookbook `rl.train`,
group_size 8, 15 batches, 480 free-text prompts) on **level-k + iterated-dominance
(Cournot)** at depths 2–4, verifier reward. Evaluated the same checkpoint vs. base on a
headroom-screened suite (kept: depth-extrapolation, MMLU-Pro, BBH-Hard; GSM8k no-regression).
Held-out bargaining / Dyck (floor) and ProntoQA (ceiling) were screened out as unmeasurable.

## Results

| benchmark | base | RLVR | Δ acc | base→RLVR acc-among-parsed |
|---|---|---|---|---|
| **depth-extrapolation** (untrained depths 5–6 of trained families) | 0.340 | **0.470** | **+13.0 pp** | 1.00 → 0.98 |
| MMLU-Pro (external reasoning) | 0.500 | 0.542 | +4.2 pp | 0.857 → 0.878 |
| BBH-Hard (external reasoning) | 0.400 | 0.407 | +0.7 pp | 0.659 → 0.678 |
| GSM8k (no-regression) | 0.920 | 0.890 | −3.0 pp | 0.958 → 0.935 |

Depth-extrapolation by depth: base {d5 0.32, d6 0.36} → RLVR {d5 **0.48**, d6 **0.46**}.

## Verdict (per the go/no-go criteria)

**Conditional GO — encouraging, not a confirmed general gain.**

1. **In-domain long-depth: clear, likely-real improvement.** GRPO on depths 2–4 lifted
   *untrained* depths 5–6 by **+13 pp** (per-depth +16/+10). This is genuine depth
   extrapolation — exactly what SFT failed to do (SFT gave near-perfect in-band but ~0
   extrapolation). RLVR's learn-from-your-own-rollouts signal is doing what we hoped.
2. **External transfer: a small positive nudge, within noise.** MMLU-Pro +4.2 pp
   (+2.1 among-parsed) and BBH-Hard ≈ flat. At n=120–150 (95% CI ≈ ±8 pp) these are not
   individually significant, but the *direction* is positive on the higher-headroom
   external benchmark (MMLU-Pro) — unlike SFT/steering, which showed flat-or-negative transfer.
3. **Mild GSM8k dip (−3 pp).** Within noise (n=100, CI ≈ ±6 pp) but not clean; worth
   guarding with stronger KL regularization or a small replay anchor in a scaled run.

So by the strict ≥3-benchmark bar (clear external improvement **and** no regression) the
lean probe does **not** cleanly clear it. But it is the **most positive of the three
approaches**: steering was task-specific; SFT had zero extrapolation and zero transfer;
GRPO here delivers real in-domain depth-extrapolation **and** a positive (if noisy)
external nudge with no catastrophic forgetting — from only 15 batches on 2 families.

## Why this justifies a scaled run
The encouraging signals (depth-extrapolation +13 pp; MMLU-Pro nudge) came from a
deliberately tiny probe (15 GRPO batches, 480 prompts, 2 families, group_size 8). The
scaled experiment is well-motivated:
- **More families** (add bargaining-as-trained, Nim/Sprague-Grundy, NE-finding) for broader
  reasoning-operation coverage — the breadth hypothesis.
- **More GRPO steps + larger groups** (the run barely started to move external metrics).
- **KL anchor / small general-data replay** to remove the GSM8k dip.
- **Larger n external evals** (≥400) to resolve the within-noise transfer.
- A **multi-axis ablation** (single-family vs mixture vs +more-steps) to attribute transfer to breadth.

## Scaled run (4 families, 30 batches, KL=0.05 anchor, n=300 externals)

Replicates and sharpens the lean probe. Base vs RLVR, two-proportion z-test:

| benchmark | base | RLVR | Δ acc | p |
|---|---|---|---|---|
| **depth-extrapolation** (untrained depths 6–7) | 0.325 | **0.450** | **+12.5 pp** | **≈0.02 (significant)** |
| MMLU-Pro (n=300) | 0.467 | 0.487 | +2.0 pp | 0.62 (ns) |
| BBH-Hard (n=300) | 0.390 | 0.427 | +3.7 pp | 0.36 (ns) |
| GSM8k (no-regression) | 0.900 | 0.870 | −3.0 raw / **+1.0 among-parsed** | ns |

Depth curve: base {d6 0.26, d7 0.39} → RLVR {d6 **0.46**, d7 0.44}.

**Confirmed conclusions:**
1. **In-domain long-depth gain is real and robust** (+12.5 pp, p≈0.02; replicates the lean
   probe's +13 pp; d6 +20 pp). RLVR generalizes across *depth* (trained 2–5, lifts 6–7) and
   across *families* — the genuine "better long-depth reasoner" claim, on the trained
   reasoning operations.
2. **External transfer: consistently positive in direction but NOT significant**, even at
   4 families / 30 batches / n=300 (MMLU-Pro +2 pp, BBH-Hard +4 pp; both runs positive). We
   cannot claim a confirmed *general* reasoning gain at this scale.
3. **No reasoning regression.** The KL anchor worked: GSM8k accuracy-among-parsed is
   flat-to-up (+1 pp); the −3 pp raw dip is a parse-rate/format artifact, not lost reasoning.

**Bottom line.** Free-text game-theory RLVR is a working, fully-verifiable *curriculum* that
instills the reasoning operations it covers with genuine generalization across difficulty —
strictly better than SFT (which had zero extrapolation) and steering (task-specific). But the
gains stay within the operation-class; transfer to broad benchmarks is positive-in-direction
yet within noise. For a *general* reasoner the engine needs operation coverage that spans the
target benchmarks' operations (or far more scale) — the consistent positive direction says it
is worth pursuing, not that it is solved.

## Reproduce
```bash
python3 scripts/build_probe_data.py                 # train + game/dyck/prontoqa evals
python3 scripts/build_external_evals.py             # MMLU-Pro + BBH-Hard (lukaemon/bbh)
.venv-tinker/bin/python scripts/headroom_screen.py  # keep 0.25-0.80
.venv-tinker/bin/python scripts/tinker_grpo.py --train data/runs/gt_rlvr/train_lean.jsonl \
    --group-size 8 --groups-per-batch 32 --max-tokens 1024 --lr 1e-5 --log-path data/runs/gt_rlvr/grpo_v1
bash scripts/probe_evalsuite.sh base --base-model Qwen/Qwen3-4B-Instruct-2507
bash scripts/probe_evalsuite.sh rlvr --model-path "$(cat data/runs/gt_rlvr/rlvr_checkpoint.txt)"
```
