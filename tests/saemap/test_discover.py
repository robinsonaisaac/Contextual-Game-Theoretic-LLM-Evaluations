import numpy as np
from game_theory_llm.saemap import discover

def test_diff_of_means_finds_planted_feature():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 20)); neg = rng.normal(size=(50, 20))
    pos[:, 7] += 3.0                      # plant signal in feature 7
    s = discover.diff_of_means(pos, neg)
    assert int(np.argmax(s)) == 7

def test_l1_probe_separates_and_selects():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 30)); y = (rng.random(120) > 0.5).astype(int)
    X[y == 1, 3] += 2.5                   # feature 3 carries the label
    res = discover.l1_probe(X, y, C=0.2)
    assert res["auc"] > 0.8
    assert 3 in res["nonzero"]

def test_decompose_direction_ranks_aligned_column():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(8, 16)); direction = W[:, 5].copy()
    ranked = discover.decompose_direction(direction, W, topn=3)
    assert ranked[0][0] == 5 and ranked[0][1] > 0.99
