"""Feature discovery: diff-of-means, L1 probe, steering-direction decomposition."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def diff_of_means(pos, neg):
    return pos.mean(0) - neg.mean(0)

def l1_probe(X, y, C=0.05, seed=0):
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())
    clf.fit(X, y)
    coef = clf.coef_[0]
    nz = np.nonzero(coef)[0]
    return {"coef": coef.tolist(), "auc": auc,
            "nonzero": nz.tolist(), "n_nonzero": int(nz.size)}

def decompose_direction(direction, W_dec, topn=20):
    d = direction / (np.linalg.norm(direction) + 1e-9)
    cols = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-9)
    cos = np.abs(d @ cols)                         # [D_SAE]
    idx = np.argsort(-cos)[:topn]
    return [(int(i), float(cos[i])) for i in idx]

def top_features(scores, k=10):
    return [int(i) for i in np.argsort(-np.abs(scores))[:k]]
