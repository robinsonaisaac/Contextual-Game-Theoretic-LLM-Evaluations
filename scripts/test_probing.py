#!/usr/bin/env python3
"""Smoke + signal test for game_theory_llm.steering.probing.

Creates synthetic ActivationBundles with a planted signal:
  - Game structure: strong cluster per game_type (SNR=3.0)
  - Context framing: weaker cluster per contrast_dim_level (SNR=1.0)

Then verifies:
  1. layer_probe on game_type  >> chance (1/7 ≈ 14%)
  2. layer_probe on contrast_dim_level  > chance (1/10 ≈ 10%)
  3. cross_framing_probe (game_type, trained on one framing, tested on another)
     generalises well — game representation is universal
  4. cross_game_context (context trained on one game, tested on another)
     generalises worse — context is game-specific
  5. game_rsa_all_layers: r_game > r_context at the signal layer
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from game_theory_llm.steering.models import ActivationBundle
from game_theory_llm.steering.probing import (
    layer_probe,
    cross_framing_probe,
    game_rsa_all_layers,
    run_full_probe_analysis,
)

# ---------------------------------------------------------------------------
# Synthetic data parameters
# ---------------------------------------------------------------------------
GAMES   = ["prisoners_dilemma", "stag_hunt", "chicken",
           "harmony", "deadlock", "battle_of_the_sexes", "matching_pennies"]
DIMS    = ["gender", "realism", "era", "contrast_domain", "observability"]
LEVELS  = {"gender": ["male", "female"],
           "realism": ["realistic", "fantasy"],
           "era": ["ancient", "modern"],
           "contrast_domain": ["business", "political"],
           "observability": ["public", "private"]}

N_LAYERS   = 8      # small for speed (real=42)
HIDDEN_DIM = 128    # small for speed (real=2560)
N_PER_CELL = 10     # stories per (game, dim, level) (real=50)

GAME_SNR    = 3.0   # signal-to-noise for game structure
CONTEXT_SNR = 0.3   # signal-to-noise for context framing (weaker)

SEED = 42


def make_bundles(rng: np.random.Generator) -> list[ActivationBundle]:
    # One fixed direction per game and per level (in the same hidden space).
    game_dirs    = {g: rng.standard_normal(HIDDEN_DIM).astype(np.float32)
                    for g in GAMES}
    level_dirs   = {}
    for dim, levels in LEVELS.items():
        for lv in levels:
            level_dirs[(dim, lv)] = rng.standard_normal(HIDDEN_DIM).astype(np.float32)

    # Layer at which the signal is strongest (simulates mid-network representation).
    SIGNAL_LAYER = N_LAYERS // 2

    bundles: list[ActivationBundle] = []
    story_idx = 0
    for game in GAMES:
        for dim, levels in LEVELS.items():
            for level in levels:
                for i in range(N_PER_CELL):
                    # Build activations for each layer.
                    activations: dict[int, dict[str, torch.Tensor]] = {}
                    for layer in range(N_LAYERS):
                        # Scale signal by proximity to SIGNAL_LAYER.
                        layer_scale = 1.0 - abs(layer - SIGNAL_LAYER) / N_LAYERS
                        noise = rng.standard_normal(HIDDEN_DIM).astype(np.float32)
                        h = (noise
                             + GAME_SNR    * layer_scale * game_dirs[game]
                             + CONTEXT_SNR * layer_scale * level_dirs[(dim, level)])
                        t = torch.tensor(h, dtype=torch.bfloat16)
                        activations[layer] = {
                            "last_prompt": t.clone(),
                            "last_trace":  t.clone(),
                            "mean_trace":  t.clone(),
                        }

                    # Cooperation: weakly correlated with game (PD → defect more)
                    coop_prob = 0.7 if game in ("harmony", "stag_hunt") else 0.4
                    cooperated = bool(rng.random() < coop_prob)

                    bundles.append(ActivationBundle(
                        story_id=f"{game}__{dim}__{level}_{i:03d}",
                        model_name="synthetic",
                        decision="A" if cooperated else "B",
                        cooperated=cooperated,
                        prompt_text="",
                        trace_text="",
                        activations=activations,
                        metadata={
                            "game_type":         game,
                            "contrast_dim":      dim,
                            "contrast_dim_level": level,
                            "cell_id":           f"{game}__{dim}__{level}",
                            "coop_choice":       "A",
                        },
                    ))
                    story_idx += 1

    return bundles


def check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)


def main():
    rng = np.random.default_rng(SEED)
    print(f"Building {len(GAMES)} games × {len(DIMS)} dims × 2 levels × {N_PER_CELL} stories "
          f"= {len(GAMES)*len(DIMS)*2*N_PER_CELL} synthetic bundles...")
    bundles = make_bundles(rng)
    print(f"  Created {len(bundles)} bundles\n")

    # ------------------------------------------------------------------
    # 1. In-distribution layer probe: game_type
    # ------------------------------------------------------------------
    print("=== 1. layer_probe(game_type) ===")
    gp = layer_probe(bundles, "game_type", position="mean_trace")
    best_game_acc = gp["test_acc"].max()
    best_game_layer = gp.loc[gp["test_acc"].idxmax(), "layer"]
    chance_game = gp["chance_acc"].iloc[0]
    print(f"  best test_acc={best_game_acc:.3f} at layer {best_game_layer}  (chance={chance_game:.3f})")
    check(best_game_acc > chance_game + 0.3, f"game_type probe well above chance")

    # ------------------------------------------------------------------
    # 2. In-distribution layer probe: contrast_dim_level
    # ------------------------------------------------------------------
    print("\n=== 2. layer_probe(contrast_dim_level) ===")
    cp = layer_probe(bundles, "contrast_dim_level", position="mean_trace")
    best_ctx_acc = cp["test_acc"].max()
    chance_ctx = cp["chance_acc"].iloc[0]
    print(f"  best test_acc={best_ctx_acc:.3f}  (chance={chance_ctx:.3f})")
    check(best_ctx_acc > chance_ctx + 0.05, f"context probe above chance")
    check(best_game_acc > best_ctx_acc, "game probe > context probe (game signal stronger)")

    # ------------------------------------------------------------------
    # 3. Cross-framing probe: game_type trained on one framing → tested on another
    # ------------------------------------------------------------------
    print("\n=== 3. cross_framing_probe(game_type across contrast_dim_level) ===")
    cfp = cross_framing_probe(
        bundles, target_label="game_type", split_label="contrast_dim_level",
        position="mean_trace",
    )
    ood = cfp[~cfp["in_distribution"]]
    mean_ood_acc = ood["test_acc"].mean()
    print(f"  cross-framing (OOD) mean acc={mean_ood_acc:.3f}  (chance={cfp['chance_acc'].iloc[0]:.3f})")
    print(f"  in-distribution mean acc={cfp[cfp['in_distribution']]['test_acc'].mean():.3f}")
    print(cfp.groupby(["train_framing", "test_framing"])["test_acc"]
             .mean().unstack().round(3).to_string())
    check(mean_ood_acc > cfp["chance_acc"].iloc[0] + 0.3,
          "game_type probe generalises cross-framing (universal representation)")

    # ------------------------------------------------------------------
    # 4. Cross-game context probe: context trained on one game → tested on another
    # ------------------------------------------------------------------
    print("\n=== 4. cross_framing_probe(contrast_dim_level across game_type) ===")
    cgc = cross_framing_probe(
        bundles, target_label="contrast_dim_level", split_label="game_type",
        position="mean_trace",
    )
    ood_ctx = cgc[~cgc["in_distribution"]]
    mean_ood_ctx = ood_ctx["test_acc"].mean()
    print(f"  cross-game (OOD) mean acc={mean_ood_ctx:.3f}  (chance={cgc['chance_acc'].iloc[0]:.3f})")

    # ------------------------------------------------------------------
    # 5. RSA: r_game vs r_context
    # ------------------------------------------------------------------
    print("\n=== 5. game_rsa_all_layers ===")
    rsa = game_rsa_all_layers(bundles, position="mean_trace")
    print(rsa[["layer", "r_game", "r_context"]].to_string(index=False))
    signal_row = rsa.loc[rsa["r_game"].idxmax()]
    check(float(signal_row["r_game"]) > float(signal_row["r_context"]),
          f"r_game > r_context at layer {int(signal_row['layer'])} "
          f"({signal_row['r_game']:.3f} > {signal_row['r_context']:.3f})")

    # ------------------------------------------------------------------
    # 6. Asymmetry check: game generalises more than context
    # ------------------------------------------------------------------
    print("\n=== 6. Universality asymmetry ===")
    print(f"  game_type cross-framing OOD acc:    {mean_ood_acc:.3f}")
    print(f"  context    cross-game   OOD acc:    {mean_ood_ctx:.3f}")
    check(mean_ood_acc > mean_ood_ctx + 0.05,
          "game representation more universal than context representation")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
