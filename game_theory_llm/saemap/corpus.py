"""Corpus builders for the decision and recognition tracks."""
from __future__ import annotations
import hashlib
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

_ELICITATION_HEADER = """\
The following is a scenario in which a character must make a decision. Read the \
scenario and then provide a brief justification and a final decision, using the \
<justification> and <decision> tags shown in the format below.

Format:
<justification>One or two sentences stating what the character decides and why.</justification>
<decision>A</decision>

---
"""

def decision_elicitation(scenario_prompt: str) -> str:
    """Prepend a concise format-header to the PD scenario prompt.

    The base model (Qwen3.5-9B-Base) sometimes enters long <think> chains
    before committing.  Prefixing a concrete one-line format example anchors
    the completion to a short <justification> + <decision> pattern, keeping
    generation within the 2000-token budget and making the final tag reliably
    findable by the judge.

    To remove cooperate-letter bias: the demonstrated decision letter in the
    format example is chosen via a stable MD5 hash of the scenario prompt,
    giving ~50/50 A vs B across any scenario set.  Option A is the cooperative
    choice in our corpus, so always demonstrating "A" in the header would
    inflate coop_rate via in-context format copying; this hash-based selection
    makes the example letter independent of the cooperative choice.
    """
    letter = "A" if (int(hashlib.md5(scenario_prompt.encode()).hexdigest(), 16) % 2 == 0) else "B"
    header = _ELICITATION_HEADER.replace("<decision>A</decision>", f"<decision>{letter}</decision>")
    return header + scenario_prompt


def elicitation_example_letter(scenario_prompt: str) -> str:
    """Return the example letter ('A' or 'B') that decision_elicitation will show for this prompt."""
    return "A" if (int(hashlib.md5(scenario_prompt.encode()).hexdigest(), 16) % 2 == 0) else "B"


def recognition_sets(n_per_game=120):
    dil = game_texts(DILEMMA_GAMES, n_per_game)
    nondil = game_texts(NONDILEMMA_GAMES, n_per_game)
    game_all = game_texts(ALL_GAMES, n_per_game)
    nong = nongame_texts(len(game_all))
    return {"dilemma": dil, "nondilemma": nondil, "game": game_all, "nongame": nong}
