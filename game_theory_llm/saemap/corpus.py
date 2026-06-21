"""Corpus builders for the decision and recognition tracks."""
from __future__ import annotations
import json
from .paths import (PD_TRAIN, PD_EVAL, STORIES_DIR, NONGAME_FILES,
                    DILEMMA_GAMES, NONDILEMMA_GAMES, ALL_GAMES)

COOP_CUE = " Decision made: I will cooperate and honor the agreement."
DEFECT_CUE = " Decision made: I will defect and break the agreement."

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def decision_pairs(path=PD_TRAIN, limit=None):
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [{"id": r["story_id"], "prompt": r["prompt"],
             "coop_letter": r["coop_choice"],
             "coop_cue": COOP_CUE, "defect_cue": DEFECT_CUE} for r in rows]

def pd_eval_set(limit=None):
    rows = load_jsonl(PD_EVAL)
    if limit is not None:
        rows = rows[:limit]
    return [{"id": r["story_id"], "prompt": r["prompt"],
             "coop_letter": r["coop_choice"]} for r in rows]

def game_texts(games, n_per_game=None):
    out = []
    for g in games:
        texts = []
        for fp in sorted(STORIES_DIR.glob(f"{g}__*.jsonl")):
            for r in load_jsonl(fp):
                texts.append(r["content"])
        out.extend(texts[:n_per_game] if n_per_game is not None else texts)
    return out

def nongame_texts(n=None):
    out = []
    for fp in NONGAME_FILES:
        if fp.exists():
            out.extend(r["prompt"] for r in load_jsonl(fp))
    return out[:n] if n is not None else out

def recognition_sets(n_per_game=120):
    dil = game_texts(DILEMMA_GAMES, n_per_game)
    nondil = game_texts(NONDILEMMA_GAMES, n_per_game)
    game_all = game_texts(ALL_GAMES, n_per_game)
    nong = nongame_texts(len(game_all))
    return {"dilemma": dil, "nondilemma": nondil, "game": game_all, "nongame": nong}
