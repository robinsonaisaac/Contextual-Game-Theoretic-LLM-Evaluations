from game_theory_llm.saemap import corpus, paths

def test_decision_pairs_have_letter_and_cues():
    rows = corpus.decision_pairs(limit=5)
    assert len(rows) == 5
    r = rows[0]
    assert r["coop_letter"] in {"A", "B"}
    assert "cooperate" in r["coop_cue"] and "defect" in r["defect_cue"]
    assert r["prompt"].strip().endswith(".")  # full prompt text present

def test_recognition_sets_partition_games():
    s = corpus.recognition_sets(n_per_game=10)
    assert len(s["dilemma"]) == 10 * len(paths.DILEMMA_GAMES)
    assert len(s["nondilemma"]) == 10 * len(paths.NONDILEMMA_GAMES)
    assert s["nongame"] and all(isinstance(t, str) for t in s["nongame"])
    # dilemma and nondilemma texts are disjoint
    assert not (set(s["dilemma"]) & set(s["nondilemma"]))
