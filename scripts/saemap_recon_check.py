"""Reconstruction validation for QwenScopeSAE at layer 16.

Extracts residuals for N=20 diverse texts (mix of dilemma / cooperation /
neutral sentences) at layer 16 via remote.extract_residuals, then prints
THREE reconstruction numbers for full transparency:

  uncentered  = 1 - ||r - rec||^2 / ||r||^2        (zero-baseline; locus validator)
  fvu_mean    = QwenScopeSAE.variance_explained(r)  (standard mean-centered FVU)
  vs_bdec     = 1 - ||r - rec||^2 / ||r - b_dec||^2 (b_dec-baseline, for reference)

Gate: asserts uncentered > 0.5. A wrong capture locus (e.g. resid_post instead
of resid_pre) gives uncentered ~0 or negative, robustly flagging misalignment
regardless of centering.
"""
import sys
import torch
import numpy as np

from game_theory_llm.saemap import remote
from game_theory_llm.saemap.sae import QwenScopeSAE

LAYER = 16

# 20 diverse texts: dilemma, cooperation, neutral, factual, reasoning
TEXTS = [
    # Prisoner's dilemma style
    "You and another player each choose: cooperate or defect. If both cooperate, you each gain 3. If you defect and they cooperate, you gain 5 and they get 0. What do you choose? <decision>",
    "In this one-shot game, mutual cooperation gives the best collective outcome, but individual defection is tempting. I choose: <decision>",
    # Cooperation / trust framing
    "Alice and Bob have worked together for years. When offered a chance to betray each other for personal gain, both choose to remain loyal. This is an example of stable cooperative equilibrium.",
    "Trust between agents can sustain cooperation even when short-run defection is tempting, provided the shadow of the future is long enough.",
    "The stag hunt dilemma: two hunters must each decide whether to hunt stag (requires coordination) or hare (safe but low payoff). Coordination yields the best outcome.",
    # Neutral / factual
    "The capital of France is Paris. The Eiffel Tower was built in 1889 and stands 330 metres tall.",
    "Water boils at 100 degrees Celsius at sea level. This is due to the balance between atmospheric pressure and the vapor pressure of water.",
    "The mitochondria are often called the powerhouse of the cell because they produce ATP through oxidative phosphorylation.",
    "Mount Everest, the highest mountain on Earth, reaches an elevation of 8,848.86 metres above sea level.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime, spanning tragedy, comedy, and history.",
    # Reasoning / math
    "If a train travels at 60 km/h for 2 hours, it covers 120 km. The time to travel 300 km at the same speed is 5 hours.",
    "To solve 2x + 5 = 17: subtract 5 from both sides to get 2x = 12, then divide by 2 to get x = 6.",
    "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8, 13 ... where each term is the sum of the two preceding ones.",
    # More game-theory / strategic
    "In a repeated prisoner's dilemma, tit-for-tat strategies can sustain cooperation if both players value future payoffs sufficiently.",
    "Nash equilibrium: a strategy profile where no player can improve their payoff by unilaterally changing their strategy, given the others' strategies.",
    "Defecting is dominant in a single-shot prisoner's dilemma, but cooperation can emerge in indefinitely repeated interactions through folk theorem mechanisms.",
    # Mixed / neutral
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once.",
    "Photosynthesis converts light energy into chemical energy stored in glucose: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.",
    "In 2024, the global average surface temperature was approximately 1.5 degrees Celsius above pre-industrial levels.",
    "A binary search algorithm finds a target value in a sorted array in O(log n) time by repeatedly halving the search space.",
]

assert len(TEXTS) == 20, f"Expected 20 texts, got {len(TEXTS)}"


def main():
    print(f"[recon_check] Extracting residuals for {len(TEXTS)} texts at layer {LAYER}...", flush=True)
    # [N, 1, D_MODEL]
    residuals_raw = remote.extract_residuals(TEXTS, [LAYER])
    print(f"[recon_check] extraction done; shape={residuals_raw.shape}", flush=True)

    # residuals_raw is [N, len(layers), D_MODEL]; squeeze layer dim -> [N, D_MODEL]
    r_np = residuals_raw[:, 0, :]   # [N, D_MODEL]
    r = torch.tensor(r_np, dtype=torch.float32)   # [N, D_MODEL]

    sae = QwenScopeSAE.load(LAYER)
    rec = sae.reconstruct(sae.encode(r))           # [N, D_MODEL]

    err_sq = (r - rec).pow(2).sum().item()

    # 1. Uncentered (zero-baseline) — robust locus validator
    zero_sq = r.pow(2).sum().item()
    uncentered = 1.0 - err_sq / zero_sq

    # 2. Standard mean-centered FVU (what variance_explained now computes)
    fvu_mean = sae.variance_explained(r)

    # 3. b_dec baseline (for reference only)
    b = sae.b_dec.to(r.device).unsqueeze(0)  # [1, D_MODEL]
    bdec_sq = (r - b).pow(2).sum().item()
    vs_bdec = 1.0 - err_sq / bdec_sq

    print()
    print(f"  uncentered  = {uncentered:.4f}   (zero-baseline; gate: must be > 0.5)")
    print(f"  fvu_mean    = {fvu_mean:.4f}   (standard mean-centered FVU)")
    print(f"  vs_bdec     = {vs_bdec:.4f}   (b_dec-baseline, reference)")
    print()

    assert uncentered > 0.5, (
        f"RECON FAIL: uncentered={uncentered:.4f} <= 0.5 — capture locus may be wrong "
        f"(expect resid_pre[L], got resid_post or misaligned layer)"
    )
    print("RECON OK")


if __name__ == "__main__":
    main()
