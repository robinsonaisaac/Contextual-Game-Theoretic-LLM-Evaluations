# examples/full_analysis.py
"""Full generation + analysis + paper-figure reproduction.

This script runs the complete pipeline described in
"Framing the Game: A Generative Approach to Contextual LLM Evaluation":

1. Generate vignettes (story generator)
2. Collect decisions from all 3 LLMs, with and without swapped A/B labels
3. Compute agreement, statistical tests, and predictive models
4. Produce every figure from the paper

Figures produced
----------------
Fig 4  - Cooperation heatmaps per model
Fig 5  - All-3-model agreement by topic × actor type
Fig 6  - Pairwise agreement matrices (two example contexts)
Fig 7  - Decision distribution original vs swapped
Fig 8  - Cooperation-proportion change when labels swapped
Fig 9c - Cramer's V per model × category
Fig 10 - Cooperation rate conditioned on game-theory recognition

Note: Figs 9a/9b (MMLU vs defection across many models) require running
experiments on additional model families and are not produced here.
Table 4 (XGBoost) requires `pip install "game_theory_llm[ml]"`.

Usage
-----
    python examples/full_analysis.py
"""

import asyncio
import json
import logging
import os
import pickle

import pandas as pd
from dotenv import load_dotenv

from game_theory_llm import (
    ALL_GAME_IDS,
    LLMClient,
    PayoffMatrix,
    StoryAnalyzer,
    StoryGenerator,
    agreement_by_context,
    classify_game_recognition,
    cooperation_by_recognition,
    fleiss_kappa,
    get_game,
    pairwise_agreement,
    plot_agreement_by_context,
    plot_cooperation_heatmaps,
    plot_cramers_v_by_model,
    plot_focal_rate_by_game,
    plot_game_recognition,
    plot_pairwise_agreement,
    plot_swap_delta_heatmaps,
    plot_swap_distribution,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

OUT_DIR = "analysis_output"


# ---------------------------------------------------------------------------
# Payoff matrix from the paper (Figure 1)
# ---------------------------------------------------------------------------
# Cooperate=A, Defect=B
#           A         B
# A   (3, 3)    (0, 5)
# B   (5, 0)    (1, 1)
PAPER_MATRIX = PayoffMatrix([
    (3, 3),  # both Cooperate
    (0, 5),  # Agent1 Cooperate, Agent2 Defect
    (5, 0),  # Agent1 Defect,    Agent2 Cooperate
    (1, 1),  # both Defect
])


async def main():
    client = LLMClient()
    generator = StoryGenerator(client)
    analyzer = StoryAnalyzer(client)

    # ------------------------------------------------------------------
    # Step 1 — Generate stories
    # ------------------------------------------------------------------
    print("Generating stories...")
    stories = await generator.generate_stories(
        payoff_matrix=PAPER_MATRIX,
        topic="mv_pharma_pro",
        actor_type="allies",
        observability="private",
        power_dynamic="symmetric",
        n_stories=20,       # increase for production runs
    )
    print(f"Generated {len(stories)} stories")

    # Persist stories for later re-use
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stories.pkl"), "wb") as f:
        pickle.dump(stories, f)

    # ------------------------------------------------------------------
    # Step 1b — Cross-game generation (new: multi-game extension)
    # ------------------------------------------------------------------
    print("\nGenerating cross-game stories (all 4 game types)...")
    cross_game_stories = []
    for game_id in ALL_GAME_IDS:
        game = get_game(game_id)
        game_stories = await generator.generate_stories(
            payoff_matrix=game.matrix,
            topic="mv_pharma_pro",
            actor_type="allies",
            observability="private",
            power_dynamic="symmetric",
            n_stories=5,
            game_config=game,
        )
        cross_game_stories.extend(game_stories)
        print(f"  {game.name}: {len(game_stories)} stories")
    print(f"Total cross-game stories: {len(cross_game_stories)}")

    # ------------------------------------------------------------------
    # Step 2 — Build analysis DataFrame (original + swapped pass)
    # ------------------------------------------------------------------
    print("\nBuilding analysis DataFrame (original + swapped)...")
    df = await analyzer.build_dataframe(stories, include_swapped=True)
    df.to_csv(os.path.join(OUT_DIR, "analysis.csv"), index=False)
    print(f"DataFrame shape: {df.shape}")

    # ------------------------------------------------------------------
    # Step 3 — Agreement metrics
    # ------------------------------------------------------------------
    print("\nComputing agreement metrics...")
    kappa = fleiss_kappa(df)
    print(f"Fleiss' Kappa: {kappa:.3f}")

    agree_df = agreement_by_context(df, group_cols=["topic", "actor_type"])
    print(agree_df.to_string(index=False))

    pw = pairwise_agreement(df)
    print("\nPairwise agreement (all stories):")
    print(pw.to_string())

    # ------------------------------------------------------------------
    # Step 4 — Paper figures
    # ------------------------------------------------------------------
    print("\nGenerating paper figures...")

    # Fig 4 — cooperation heatmaps
    plot_cooperation_heatmaps(df, save_dir=OUT_DIR)

    # Fig 5 — agreement by context
    plot_agreement_by_context(agree_df, save_dir=OUT_DIR)

    # Fig 6 — pairwise agreement for two contrasting contexts
    #          (adapt topic/actor to your actual data)
    for topic, actor in [("mv_pharma_pro", "allies")]:
        pw_ctx = pairwise_agreement(df, topic=topic, actor_type=actor)
        plot_pairwise_agreement(
            pw_ctx,
            context_label=f"{actor} × {topic}",
            save_dir=OUT_DIR,
        )

    # Fig 7 — swap distribution
    plot_swap_distribution(df, save_dir=OUT_DIR)

    # Fig 8 — swap delta heatmaps
    plot_swap_delta_heatmaps(df, save_dir=OUT_DIR)

    # Fig 9c — Cramer's V
    plot_cramers_v_by_model(df, save_dir=OUT_DIR)

    # Cross-game focal rate comparison
    if "game_type" in df.columns and df["game_type"].nunique() > 1:
        plot_focal_rate_by_game(df, save_dir=OUT_DIR)

    # ------------------------------------------------------------------
    # Step 5 — Game-recognition analysis (Fig 10, Appendix E)
    # Uses Claude to classify whether justifications mention game theory.
    # ------------------------------------------------------------------
    print("\nClassifying game-theory recognition...")
    # Use the llama justification column (adjust as needed)
    if "response_llama" in df.columns:
        df["game_theory_mentioned"] = await classify_game_recognition(
            df, client, justification_col="response_llama"
        )
        rec_df = cooperation_by_recognition(df)
        plot_game_recognition(rec_df, save_dir=OUT_DIR)
        print("Game-recognition results:")
        print(rec_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 6 — XGBoost predictive models (Table 4, Appendix C)
    #           Requires: pip install "game_theory_llm[ml]"
    # ------------------------------------------------------------------
    try:
        from game_theory_llm import CategoricalPredictor, EmbeddingPredictor

        print("\nTraining XGBoost predictors (Table 4)...")
        cat_results = CategoricalPredictor().fit(df).evaluate()
        print("Table 4a — Categorical predictor:")
        print(cat_results.round(2).to_string())

        emb_results = EmbeddingPredictor().fit(df).evaluate()
        print("\nTable 4b — Embedding predictor:")
        print(emb_results.round(2).to_string())

        cat_results.to_csv(os.path.join(OUT_DIR, "table4a_categorical.csv"))
        emb_results.to_csv(os.path.join(OUT_DIR, "table4b_embedding.csv"))

    except ImportError:
        print(
            "\nSkipping XGBoost predictors — run:\n"
            '  pip install "game_theory_llm[ml]"\n'
            "to enable predictive analysis."
        )

    print(f"\nAll outputs saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    asyncio.run(main())
