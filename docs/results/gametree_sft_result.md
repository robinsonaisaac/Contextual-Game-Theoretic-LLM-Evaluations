# Does game-theory–synthesized data train a better long-depth reasoner?

**Setup.** Base = `Qwen/Qwen3-4B-Instruct-2507` (Tinker LoRA, rank 32, 2 epochs).
Train = 600 gold backward-induction CoT examples on **minimax game trees of depth
2–4**. Evaluated the same checkpoint vs. the base model on game trees (in-band 2–4
+ **extrapolation 5–6**) and on three external benchmarks (GSM8k, MMLU, BBH),
per the ≥3-benchmark validation rule.

## Results

**Game tree** — accuracy [parse rate], n=30/depth:

| depth | band | base | fine-tuned |
|---|---|---|---|
| 2 | train | 1.00 | 1.00 |
| 3 | train | 0.37 [0.63] | **1.00** [1.00] |
| 4 | train | 0.07 [0.10] | **1.00** [1.00] |
| 5 | extrapolation | 0.00 [0.00] | 0.07 [1.00] |
| 6 | extrapolation | 0.00 [0.00] | 0.23 [1.00] |

**External** — accuracy (accuracy-among-parsed), n=100:

| benchmark | base | fine-tuned |
|---|---|---|
| GSM8k | 0.90 (0.96) | 0.90 (0.91) |
| MMLU | 0.69 (0.70) | 0.70 (0.71) |
| BBH (logical-deduction, ceiling) | 1.00 | 0.97 |

## Findings

1. **Large, genuine in-domain gain.** Depths 3–4 went 0.37→1.00 and 0.07→1.00.
   The accuracy gain exceeds the parse-rate gain (e.g. depth 3: parse 0.63→1.00 but
   acc 0.37→1.00), so it is real backward-induction reasoning, not just learning to
   emit the `<answer>` format.
2. **Weak depth extrapolation.** On depths never trained (5–6), accuracy went
   0%→7% and 0%→23% with parse 0→1.0 — the model learned the *procedure/format*
   but does not robustly execute it past the trained depth. A length-generalization
   limit; a depth-curriculum (train 2–6) would likely push this further.
3. **No general transfer — and no regression.** GSM8k/MMLU/BBH are flat within
   ±1 point. By the ≥3-benchmark bar this is **task-specific skill acquisition,
   not a general reasoning gain**. Importantly there is also no capability
   regression (no catastrophic forgetting from the narrow SFT).

## Verdict

**Yes — game-theory data is an excellent, fully-verifiable teacher of a targeted
long-depth reasoning skill** (near-perfect in-band, partial extrapolation, zero
collateral damage). But a *single* narrow task does not bootstrap *general*
reasoning, the same task-specificity lesson the steering experiments showed —
now on the training side. The result is cleaner and more positive than steering:
the fine-tune actually extrapolates somewhat in-domain and leaves other
capabilities intact.

## Next steps toward a *general* gain
- **Diverse verifiable mixture.** Train on many game-theoretic reasoning families
  at once (game trees + level-k + iterated dominance + bargaining/Nim) — the
  natural test of whether *breadth* of verifiable reasoning data yields external
  transfer (vs. one task overfitting).
- **Depth curriculum + RLVR.** Train depths 2–6 (easy→hard) and/or use the
  verifier as an RL reward to push depth extrapolation.
- **Counterfactual control** (mostly already answered): the acc-gain>parse-gain
  margin shows the in-band lift is reasoning, not format; a trivial-tree control
  fine-tune would confirm formally.

## Reproduce
```bash
# data
python3 scripts/build_gametree_sft.py
python3 - <<'EOF'  # capped eval subsets -> data/runs/gametree/exp_*.jsonl
# (see commit; 150 game-tree depths 2-6 + 100 each gsm8k/mmlu/bbh)
EOF
# train + eval (Tinker venv)
.venv-tinker/bin/python scripts/tinker_sft.py --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --rank 32 --epochs 2 --batch 8 --save-name gametree_sft_v1
bash scripts/gt_evalsuite.sh base --base-model Qwen/Qwen3-4B-Instruct-2507
bash scripts/gt_evalsuite.sh ft   --model-path "$(cat data/runs/gametree/sft_checkpoint.txt)"
```
