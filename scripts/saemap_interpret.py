"""Driver: interpret shortlisted SAE features by max-activating examples.

Reads discovery summary + per-contrast JSON files, builds an interp pool,
fetches residuals remotely (one call per needed layer), encodes locally,
and writes:
  data/runs/saemap_9b/interpret/top_features.json

Optionally labels each feature with a short human-readable name via an
LLM judge (Claude Sonnet via OpenRouter), as required by project rules
(no regex for quality assessment).

Usage:
    cd /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering
    source ../../.env
    .venv-sae/bin/python scripts/saemap_interpret.py [--no-label]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure package root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_theory_llm.saemap import corpus, paths, remote
from game_theory_llm.saemap.interpret import batch_max_activating
from game_theory_llm.saemap.sae import QwenScopeSAE

# ---------------------------------------------------------------------------
# Which features to interpret, per the task-8 brief.
# These are the MEANINGFUL features (nonzero probe coef + agreed diff-of-means)
# ---------------------------------------------------------------------------
FEATURES_OF_INTEREST = {
    # decision @ L8: feature 2448 is the single nonzero probe feature AND
    # the #1 diff-of-means feature.  Also grab next two diff-of-means for context.
    "decision": {
        "layer": 8,
        "features": [2448, 21511, 38905],
    },
    # recog_fine @ L24: agreed by both probe and diff-of-means
    "recog_fine": {
        "layer": 24,
        "features": [48983, 29366, 14407, 51436, 16610],
    },
    # recog_coarse @ L8: single nonzero, plus #2 diff-of-means
    "recog_coarse": {
        "layer": 8,
        "features": [50944, 3618],
    },
}

TOPN = 8   # examples per feature


def build_pool() -> list[str]:
    """Build the interpretation pool (≈ 320 texts, no forced cue words)."""
    game_texts = corpus.game_texts(
        paths.DILEMMA_GAMES + paths.NONDILEMMA_GAMES, 40
    )
    nongame_texts = corpus.nongame_texts(120)
    pool = game_texts + nongame_texts
    print(f"  Interp pool: {len(game_texts)} game + {len(nongame_texts)} nongame = {len(pool)} total")
    return pool


# ---------------------------------------------------------------------------
# Optional LLM labelling (Claude Sonnet via OpenRouter)
# ---------------------------------------------------------------------------
_LABEL_SYSTEM = (
    "You are a neural-network interpretability researcher. "
    "You will be shown up to 8 short text snippets that most strongly activate "
    "a sparse autoencoder feature in a language model. "
    "Your task is to propose a concise (3-7 word) human-readable label for what "
    "this feature seems to represent, based solely on the common theme across snippets. "
    "Respond with ONLY a JSON object: {\"label\": \"...\"}. No prose."
)


def _label_feature(client, snippets: list[dict]) -> str:
    """Ask Sonnet to propose a label for a feature from its top snippets."""
    lines = []
    for i, s in enumerate(snippets[:8], 1):
        lines.append(f"Snippet {i} (score={s['score']:.3f}):\n{s['snippet']}")
    user_msg = "Top-activating text snippets:\n\n" + "\n\n---\n\n".join(lines)
    try:
        resp = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-6",
            temperature=0.0,
            messages=[
                {"role": "system", "content": _LABEL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=64,
        )
        raw = resp.choices[0].message.content or ""
        parsed = json.loads(raw)
        return parsed.get("label", "unlabeled")
    except Exception as e:
        print(f"    Warning: labeling failed: {e}")
        return "unlabeled"


def label_features(features_out: dict) -> dict:
    """Add a 'label' key to each feature entry using LLM judge."""
    import openai
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  Warning: OPENROUTER_API_KEY not set — skipping LLM labels")
        return features_out

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    labeled = {}
    for contrast, feat_dict in features_out.items():
        labeled[contrast] = {}
        for fid_str, snippets in feat_dict.items():
            print(f"    Labeling {contrast}/feature {fid_str} …")
            lbl = _label_feature(client, snippets)
            labeled[contrast][fid_str] = {
                "label": lbl,
                "examples": snippets,
            }
            print(f"      -> \"{lbl}\"")
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-label", action="store_true",
                        help="Skip LLM labelling (just write snippets)")
    args = parser.parse_args()

    paths.ensure_run_dirs()

    # Build pool once
    print("Building interpretation pool …")
    pool = build_pool()

    # Group features by layer so we batch ONE remote call per layer
    layers_needed = {}  # layer -> {contrast: [feature_ids]}
    for contrast, info in FEATURES_OF_INTEREST.items():
        L = info["layer"]
        layers_needed.setdefault(L, {})[contrast] = info["features"]

    features_out: dict[str, dict] = {c: {} for c in FEATURES_OF_INTEREST}

    for L, contrast_map in sorted(layers_needed.items()):
        print(f"\nFetching residuals at layer {L} (one remote call) …")
        # Collect all features for this layer across contrasts
        all_features = []
        for feats in contrast_map.values():
            all_features.extend(feats)
        all_features = sorted(set(all_features))

        print(f"  Loading SAE for layer {L} …")
        sae = QwenScopeSAE.load(L)

        print(f"  Calling remote.extract_residuals for {len(pool)} texts …")
        feat_results = batch_max_activating(L, all_features, pool, sae, topn=TOPN)

        for contrast, feats in contrast_map.items():
            for f in feats:
                features_out[contrast][str(f)] = feat_results[f]
                top_score = feat_results[f][0]["score"] if feat_results[f] else 0.0
                print(f"  {contrast}/feature {f}: top score = {top_score:.4f}")

    # Optionally run LLM labeling
    if not args.no_label:
        print("\nRunning LLM labeling (Sonnet via OpenRouter) …")
        features_out = label_features(features_out)

    out_path = paths.RUN_DIR / "interpret" / "top_features.json"
    out_path.write_text(json.dumps(features_out, indent=2))
    print(f"\nWrote {out_path}")
    print("INTERPRET DONE")


if __name__ == "__main__":
    main()
