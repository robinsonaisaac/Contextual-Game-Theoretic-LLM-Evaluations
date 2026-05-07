"""Offline mechanistic interpretability analysis for the game-framing project.

All functions here run on CPU from saved .pt bundles — no GPU needed after
activation extraction.

Core claim
----------
The model maintains a *universal* representation of game structure — stories
about the same game cluster together in activation space regardless of surface
framing.  Context lives in the residual variation around that structure.

Key experiment: cross_framing_probe()
    Train a game_type probe on stories with framing A (e.g. gender__male).
    Test it on stories with framing B (e.g. gender__female).
    High cross-framing accuracy → game representation is universal.
    Compare: context probe trained on PD stories, tested on Stag Hunt stories.
    If game generalises cross-framing but context does NOT generalise cross-game
    → the asymmetry is the mechanistic result.

Other experiments
-----------------
layer_probe()            Standard in-distribution probing per layer (E1)
logit_attribution()      Per-layer A-vs-B logit contribution (E2)
game_rsa()               Representational similarity: does sim-matrix align with
                         game identity more than framing identity? (RSA)
subspace_decomposition() Cosine sim between game and context directions (E4)
residual_probe()         Probing after projecting out sentence-embedding subspace
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .models import ActivationBundle
from .storage import load_activation_bundle, read_index


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_index_and_bundles(
    run_dir: Path | str,
    split: str = "train",
    require_meta: bool = True,
) -> tuple[pd.DataFrame, list[ActivationBundle]]:
    """Load index and all activation bundles for a run.

    Parameters
    ----------
    run_dir : path to the run directory (contains index.parquet + activations/)
    split   : "train" or "eval"
    require_meta : if True, skip bundles missing game_type / contrast_dim metadata

    Returns (index_df, bundles) where index_df has one row per bundle.
    """
    run_dir = Path(run_dir)
    index_df = read_index(run_dir / "index.parquet", split=split)

    bundles: list[ActivationBundle] = []
    kept_rows: list[int] = []
    for i, row in index_df.iterrows():
        # Remap volume path to local path if needed.
        path_str = row["path"]
        if path_str.startswith("/data/runs/"):
            rel = Path(path_str).relative_to("/data/runs")
            # rel = "{run_id}/activations/{split}/{story_id}.pt"
            # strip the run_id prefix
            parts = rel.parts
            local_path = run_dir / Path(*parts[1:])
        else:
            local_path = Path(path_str)
        if not local_path.exists():
            continue
        b = load_activation_bundle(local_path)
        if require_meta and not b.metadata.get("game_type"):
            continue
        bundles.append(b)
        kept_rows.append(i)

    return index_df.loc[kept_rows].reset_index(drop=True), bundles


def _get_label(bundle: ActivationBundle, label_col: str) -> str | None:
    """Extract a label from bundle metadata or top-level fields."""
    if label_col == "cooperated":
        return str(bundle.cooperated)
    if label_col == "decision":
        return bundle.decision or None
    return bundle.metadata.get(label_col)


def _stack_layer(bundles: list[ActivationBundle], layer: int,
                 position: str) -> torch.Tensor:
    """Stack activations for (layer, position) across bundles → (n, hidden_dim) float32."""
    vecs = []
    for b in bundles:
        if layer in b.activations and position in b.activations[layer]:
            vecs.append(b.activations[layer][position].float())
    if not vecs:
        raise ValueError(f"No activations found for layer={layer}, position={position}")
    return torch.stack(vecs)


# ---------------------------------------------------------------------------
# E1 — Layer-by-layer linear probing
# ---------------------------------------------------------------------------

def layer_probe(
    bundles: list[ActivationBundle],
    label_col: str,
    position: str = "mean_trace",
    test_frac: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Train a linear probe at each layer to predict *label_col* from activations.

    Parameters
    ----------
    bundles   : list of ActivationBundle from load_index_and_bundles()
    label_col : metadata key to predict ("game_type", "contrast_dim_level",
                "contrast_dim", "cooperated", ...)
    position  : which residual-stream position to use ("mean_trace" recommended)
    test_frac : held-out fraction for accuracy evaluation
    seed      : random seed for train/test split

    Returns a DataFrame with columns [layer, label_col, n_classes, n_train, n_test,
    train_acc, test_acc].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Filter to bundles that have the label.
    valid = [(b, _get_label(b, label_col)) for b in bundles]
    valid = [(b, lbl) for b, lbl in valid if lbl is not None]
    if len(valid) < 10:
        raise ValueError(f"Too few bundles with label '{label_col}': {len(valid)}")

    labels_raw = [lbl for _, lbl in valid]
    valid_bundles = [b for b, _ in valid]
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    # Discover layers from first bundle.
    layers = sorted(valid_bundles[0].activations.keys())
    rows = []
    for layer in layers:
        try:
            X = _stack_layer(valid_bundles, layer, position).numpy()
        except ValueError:
            continue

        X = X.astype(np.float64)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_frac, random_state=seed, stratify=y if len(le.classes_) > 1 else None
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        np.clip(X_tr, -50, 50, out=X_tr)
        np.clip(X_te, -50, 50, out=X_te)
        clf = LogisticRegression(max_iter=1000, C=0.1, solver="liblinear")
        clf.fit(X_tr, y_tr)
        rows.append({
            "layer": layer,
            "label_col": label_col,
            "position": position,
            "n_classes": len(le.classes_),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "train_acc": clf.score(X_tr, y_tr),
            "test_acc": clf.score(X_te, y_te),
            "chance_acc": 1.0 / len(le.classes_),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E2 — Logit attribution
# ---------------------------------------------------------------------------

def logit_attribution(
    bundles: list[ActivationBundle],
    W_U: torch.Tensor,
    a_token_id: int,
    b_token_id: int,
    position: str = "mean_trace",
) -> pd.DataFrame:
    """Compute per-layer contribution to the A-vs-B logit difference.

    Uses the direct logit attribution decomposition:
        logit_diff_l = (W_U[A_id] - W_U[B_id]) · delta_h_l
    where delta_h_l = h_l - h_{l-1} is the residual update at layer l.

    W_U should be the lm_head weight matrix: shape (vocab_size, hidden_dim), float32.
    Obtain it via `SteeringWorker.get_unembedding.remote()`.

    Returns a DataFrame with columns [layer, position, mean_attr, std_attr, n].
    mean_attr > 0 means layer l pushes toward A; < 0 toward B.
    """
    W_U = W_U.float()
    direction = (W_U[a_token_id] - W_U[b_token_id]).float()  # (hidden_dim,)
    direction = direction / (direction.norm() + 1e-8)

    layers = sorted(bundles[0].activations.keys())

    # Build h matrix per layer: shape (n_bundles, hidden_dim)
    layer_stacks: dict[int, torch.Tensor] = {}
    for layer in layers:
        try:
            layer_stacks[layer] = _stack_layer(bundles, layer, position)
        except ValueError:
            pass

    rows = []
    prev = None
    for layer in sorted(layer_stacks):
        h = layer_stacks[layer].float()
        if prev is None:
            delta = h  # layer 0 contribution is the full embedding
        else:
            delta = h - prev
        prev = h

        # Project each story's delta onto the A-B direction.
        attrs = delta @ direction   # (n_bundles,)
        rows.append({
            "layer": layer,
            "position": position,
            "mean_attr": float(attrs.mean()),
            "std_attr": float(attrs.std()),
            "n": len(attrs),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E4 — Subspace decomposition
# ---------------------------------------------------------------------------

def _mean_direction(
    bundles: list[ActivationBundle],
    layer: int,
    position: str,
    label_col: str,
    class_a: str,
    class_b: str,
) -> torch.Tensor:
    """Compute mean(class_a) - mean(class_b) at (layer, position), unit-normalised."""
    a_acts = [b.activations[layer][position].float()
               for b in bundles if _get_label(b, label_col) == class_a
               and layer in b.activations and position in b.activations[layer]]
    b_acts = [b.activations[layer][position].float()
               for b in bundles if _get_label(b, label_col) == class_b
               and layer in b.activations and position in b.activations[layer]]
    if not a_acts or not b_acts:
        raise ValueError(f"Not enough examples for {label_col}: {class_a}={len(a_acts)}, {class_b}={len(b_acts)}")
    diff = torch.stack(a_acts).mean(0) - torch.stack(b_acts).mean(0)
    return diff / (diff.norm() + 1e-8)


def subspace_decomposition(
    bundles: list[ActivationBundle],
    label_a: str,
    class_a1: str,
    class_a2: str,
    label_b: str,
    class_b1: str,
    class_b2: str,
    position: str = "mean_trace",
) -> pd.DataFrame:
    """Measure cosine similarity between two contrastive directions per layer.

    Typical use: label_a="game_type", class_a1="prisoners_dilemma", class_a2="harmony"
                 label_b="contrast_dim_level", class_b1="male", class_b2="female"

    Returns DataFrame with columns [layer, cosine_sim, raw_norm_a, raw_norm_b].
    cosine_sim ≈ 0 → directions are orthogonal (cleanly separated representations).
    cosine_sim ≈ 1 → directions are aligned (entangled representations).
    """
    layers = sorted(bundles[0].activations.keys())
    rows = []
    for layer in layers:
        try:
            dir_a = _mean_direction(bundles, layer, position, label_a, class_a1, class_a2)
            dir_b = _mean_direction(bundles, layer, position, label_b, class_b1, class_b2)
        except ValueError:
            continue

        # Also compute raw (unnormalised) norms to measure representational strength.
        a_acts_all = [b.activations[layer][position].float()
                      for b in bundles if _get_label(b, label_a) in (class_a1, class_a2)
                      and layer in b.activations]
        b_acts_all = [b.activations[layer][position].float()
                      for b in bundles if _get_label(b, label_b) in (class_b1, class_b2)
                      and layer in b.activations]

        def _raw_norm(bundles_subset, label, c1, c2):
            a = [x.float() for b in bundles_subset
                 for x in [b.activations[layer][position]]
                 if _get_label(b, label) == c1]
            b = [x.float() for bun in bundles_subset
                 for x in [bun.activations[layer][position]]
                 if _get_label(bun, label) == c2]
            if not a or not b:
                return float("nan")
            return float((torch.stack(a).mean(0) - torch.stack(b).mean(0)).norm())

        rows.append({
            "layer": layer,
            "position": position,
            "cosine_sim": float((dir_a * dir_b).sum().abs()),
            "raw_norm_a": _raw_norm(bundles, label_a, class_a1, class_a2),
            "raw_norm_b": _raw_norm(bundles, label_b, class_b1, class_b2),
        })

    return pd.DataFrame(rows)


def direction_decision_correlation(
    bundles: list[ActivationBundle],
    label_col: str,
    class_a: str,
    class_b: str,
    position: str = "mean_trace",
) -> pd.DataFrame:
    """Measure how well projecting onto the contrastive direction predicts cooperation.

    For each layer: project each bundle's activation onto the (class_a - class_b)
    direction, then compute Pearson correlation with the cooperated flag.

    Returns DataFrame with columns [layer, pearson_r, n].
    High |pearson_r| → this direction is predictive of cooperation.
    """
    layers = sorted(bundles[0].activations.keys())
    y = np.array([float(b.cooperated) for b in bundles])
    rows = []
    for layer in layers:
        try:
            classes = list({_get_label(b, label_col) for b in bundles
                            if _get_label(b, label_col) is not None})
            if class_a not in classes or class_b not in classes:
                continue
            direction = _mean_direction(bundles, layer, position, label_col, class_a, class_b)
            X = _stack_layer(bundles, layer, position).float()
            proj = (X @ direction).numpy()
            r = float(np.corrcoef(proj, y)[0, 1])
        except (ValueError, np.linalg.LinAlgError):
            r = float("nan")

        rows.append({"layer": layer, "position": position,
                     "pearson_r": r, "n": len(bundles)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Residual probing (Option B — controls for surface text)
# ---------------------------------------------------------------------------

def residual_probe(
    bundles: list[ActivationBundle],
    label_col: str,
    sentence_embeddings: np.ndarray,
    position: str = "mean_trace",
    test_frac: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Probe activations after projecting out the sentence-embedding subspace.

    sentence_embeddings : (n_bundles, emb_dim) float32 array, same order as bundles.
    Residual = activation - (activation · e_hat) * e_hat
    where e_hat is the unit sentence-embedding direction for that story.

    Returns same schema as layer_probe(). Compare test_acc here to layer_probe()
    to see how much probe accuracy comes from surface text vs. abstract representation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    valid = [(i, b, _get_label(b, label_col))
             for i, b in enumerate(bundles)]
    valid = [(i, b, lbl) for i, b, lbl in valid if lbl is not None]
    if len(valid) < 10:
        raise ValueError(f"Too few bundles with label '{label_col}': {len(valid)}")

    indices = [i for i, _, _ in valid]
    valid_bundles = [b for _, b, _ in valid]
    labels_raw = [lbl for _, _, lbl in valid]
    embs = torch.tensor(sentence_embeddings[indices], dtype=torch.float32)
    emb_norms = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)  # (n, emb_dim)

    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    layers = sorted(valid_bundles[0].activations.keys())
    rows = []
    for layer in layers:
        try:
            X = _stack_layer(valid_bundles, layer, position).float()
        except ValueError:
            continue

        # Project out sentence embedding direction per story.
        # residual_i = x_i - (x_i · e_i) * e_i   (e_i already unit-normalised)
        # We need to handle emb_dim != hidden_dim; use a projection matrix instead.
        # Simple approach: subtract the component along the first PC of embs.
        # Full approach: project onto null space of embs (more expensive).
        # Here we use the simple per-story projection.
        if emb_norms.shape[1] != X.shape[1]:
            # Dimensionality mismatch: fit a linear map emb → hidden_dim and project.
            # Use the pseudo-inverse: P = embs_pinv @ X, then residual = X - embs @ P
            import torch.linalg as tla
            P = tla.lstsq(embs, X).solution   # (emb_dim, hidden_dim)
            pred = embs @ P
            X_res = X - pred
        else:
            proj_scalars = (X * emb_norms).sum(dim=1, keepdim=True)  # (n, 1)
            X_res = X - proj_scalars * emb_norms

        X_np = X_res.numpy().astype(np.float64)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_np, y, test_size=test_frac, random_state=seed,
            stratify=y if len(le.classes_) > 1 else None,
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        np.clip(X_tr, -50, 50, out=X_tr)
        np.clip(X_te, -50, 50, out=X_te)
        clf = LogisticRegression(max_iter=1000, C=0.1, solver="liblinear")
        clf.fit(X_tr, y_tr)
        rows.append({
            "layer": layer,
            "label_col": label_col,
            "position": position,
            "n_classes": len(le.classes_),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "train_acc": clf.score(X_tr, y_tr),
            "test_acc": clf.score(X_te, y_te),
            "chance_acc": 1.0 / len(le.classes_),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-framing probe generalization — the core universality test
# ---------------------------------------------------------------------------

def cross_framing_probe(
    bundles: list[ActivationBundle],
    target_label: str = "game_type",
    split_label: str = "contrast_dim_level",
    position: str = "mean_trace",
    best_layer: int | None = None,
) -> pd.DataFrame:
    """Test whether a game_type probe trained on one framing generalises to another.

    For every pair of (train_framing, test_framing) where framing = a unique
    value of *split_label* (e.g. "male", "female", "ancient", "modern", ...):
      - train logistic regression on bundles where split_label == train_framing
      - test on bundles where split_label == test_framing (held-out)

    If *best_layer* is None, the function first runs an in-distribution probe
    at every layer and uses the layer with highest mean test_acc.

    Returns a DataFrame with columns:
        train_framing, test_framing, layer, n_train, n_test, test_acc, chance_acc
        in_distribution (bool: True when train_framing == test_framing)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Filter to bundles that have both labels.
    valid = [
        b for b in bundles
        if _get_label(b, target_label) is not None
        and _get_label(b, split_label) is not None
    ]
    if len(valid) < 10:
        raise ValueError(f"Too few bundles with both '{target_label}' and '{split_label}'")

    framings = sorted({_get_label(b, split_label) for b in valid})
    layers = sorted(valid[0].activations.keys())

    # If no best_layer given, find it via in-distribution probe.
    if best_layer is None:
        # Use all data (no cross-framing split) to find the best layer.
        le_tmp = LabelEncoder()
        y_all = le_tmp.fit_transform([_get_label(b, target_label) for b in valid])
        best_acc, best_layer = 0.0, layers[0]
        for layer in layers:
            try:
                X = _stack_layer(valid, layer, position).numpy().astype(np.float64)
            except ValueError:
                continue
            from sklearn.model_selection import cross_val_score
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, C=0.1, solver="liblinear")),
                    ]),
                    X, y_all, cv=3,
                )
            acc = float(scores.mean())
            if acc > best_acc:
                best_acc, best_layer = acc, layer
        print(f"[cross_framing_probe] best layer for '{target_label}': {best_layer} "
              f"(in-dist acc={best_acc:.3f})", flush=True)

    rows = []
    le = LabelEncoder()
    le.fit([_get_label(b, target_label) for b in valid])
    chance = 1.0 / len(le.classes_)

    for train_framing in framings:
        train_b = [b for b in valid if _get_label(b, split_label) == train_framing]
        if len(train_b) < 5:
            continue
        try:
            X_tr = _stack_layer(train_b, best_layer, position).numpy().astype(np.float64)
        except ValueError:
            continue
        y_tr = le.transform([_get_label(b, target_label) for b in train_b])

        # Only fit if we have at least 2 classes in training set.
        if len(set(y_tr)) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        np.clip(X_tr, -50, 50, out=X_tr)
        clf = LogisticRegression(max_iter=1000, C=0.1, solver="liblinear")
        clf.fit(X_tr, y_tr)

        for test_framing in framings:
            test_b = [b for b in valid if _get_label(b, split_label) == test_framing]
            if len(test_b) < 3:
                continue
            try:
                X_te = _stack_layer(test_b, best_layer, position).numpy().astype(np.float64)
            except ValueError:
                continue
            X_te = scaler.transform(X_te)
            np.clip(X_te, -50, 50, out=X_te)
            y_te = le.transform([_get_label(b, target_label) for b in test_b])
            # Skip if test set only has one class (can't evaluate meaningfully).
            if len(set(y_te)) < 2:
                continue
            rows.append({
                "train_framing": train_framing,
                "test_framing": test_framing,
                "layer": best_layer,
                "n_train": len(train_b),
                "n_test": len(test_b),
                "test_acc": float(clf.score(X_te, y_te)),
                "chance_acc": chance,
                "in_distribution": train_framing == test_framing,
                "target_label": target_label,
                "split_label": split_label,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Representational Similarity Analysis — does activation similarity align
# with game identity more than framing identity?
# ---------------------------------------------------------------------------

def game_rsa(
    bundles: list[ActivationBundle],
    layer: int,
    position: str = "mean_trace",
) -> dict[str, float]:
    """Compute Spearman correlation of the pairwise activation similarity matrix
    with two "ground truth" similarity matrices:

      S_game[i,j]    = 1 if same game_type,          else 0
      S_context[i,j] = 1 if same contrast_dim_level, else 0

    Returns {"r_game": float, "r_context": float, "n_pairs": int}.

    r_game > r_context → game identity organises the representation more than
    framing → evidence of a universal game-structure representation.
    """
    from scipy.stats import spearmanr

    valid = [b for b in bundles
             if _get_label(b, "game_type") is not None
             and _get_label(b, "contrast_dim_level") is not None]
    if len(valid) < 10:
        raise ValueError("Too few bundles for RSA")

    try:
        X = _stack_layer(valid, layer, position).float().numpy()
    except ValueError as e:
        raise ValueError(f"RSA failed at layer {layer}: {e}")

    n = len(valid)
    # Pairwise cosine similarity (upper triangle only).
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = X_norm @ X_norm.T  # (n, n)

    game_labels = np.array([_get_label(b, "game_type") for b in valid])
    ctx_labels  = np.array([_get_label(b, "contrast_dim_level") for b in valid])

    idx = np.triu_indices(n, k=1)
    sim_vec   = sim[idx]
    game_vec  = (game_labels[idx[0]] == game_labels[idx[1]]).astype(float)
    ctx_vec   = (ctx_labels[idx[0]]  == ctx_labels[idx[1]]).astype(float)

    r_game,    _ = spearmanr(sim_vec, game_vec)
    r_context, _ = spearmanr(sim_vec, ctx_vec)

    return {
        "layer": layer,
        "position": position,
        "r_game": float(r_game),
        "r_context": float(r_context),
        "n": n,
        "n_pairs": len(sim_vec),
    }


def game_rsa_all_layers(
    bundles: list[ActivationBundle],
    position: str = "mean_trace",
) -> pd.DataFrame:
    """Run game_rsa() at every layer. Returns a DataFrame."""
    layers = sorted(bundles[0].activations.keys())
    rows = []
    for layer in layers:
        try:
            rows.append(game_rsa(bundles, layer, position))
        except ValueError:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined analysis runner
# ---------------------------------------------------------------------------

def run_full_probe_analysis(
    bundles: list[ActivationBundle],
    W_U: Optional[torch.Tensor] = None,
    a_token_id: Optional[int] = None,
    b_token_id: Optional[int] = None,
    position: str = "mean_trace",
) -> dict[str, pd.DataFrame]:
    """Run the full universality + attribution analysis in one call.

    Returns a dict with keys:
      "game_probe"          layer_probe on game_type (in-distribution)
      "context_probe"       layer_probe on contrast_dim_level
      "coop_probe"          layer_probe on cooperated
      "rsa"                 game_rsa_all_layers — r_game vs r_context per layer
      "cross_framing_game"  cross_framing_probe for game_type across framings
      "cross_game_context"  cross_framing_probe for contrast_dim_level across games
      "logit_attr"          logit_attribution (if W_U + token IDs provided)
      "subspace"            subspace_decomposition (PD vs Harmony × male vs female)
    """
    results = {}

    print("[probe] running game_type probe (in-distribution)...", flush=True)
    results["game_probe"] = layer_probe(bundles, "game_type", position=position)

    print("[probe] running contrast_dim_level probe...", flush=True)
    try:
        results["context_probe"] = layer_probe(bundles, "contrast_dim_level", position=position)
    except ValueError as e:
        print(f"[probe] context_probe skipped: {e}")

    print("[probe] running cooperated probe...", flush=True)
    results["coop_probe"] = layer_probe(bundles, "cooperated", position=position)

    print("[probe] running RSA (r_game vs r_context per layer)...", flush=True)
    results["rsa"] = game_rsa_all_layers(bundles, position=position)

    # Core universality test: does game_type probe generalise cross-framing?
    print("[probe] running cross-framing probe (game_type across contrast_dim_level)...", flush=True)
    try:
        results["cross_framing_game"] = cross_framing_probe(
            bundles, target_label="game_type", split_label="contrast_dim_level",
            position=position,
        )
    except ValueError as e:
        print(f"[probe] cross_framing_game skipped: {e}")

    # Contrast: does context probe generalise cross-game?
    print("[probe] running cross-game probe (contrast_dim_level across game_type)...", flush=True)
    try:
        results["cross_game_context"] = cross_framing_probe(
            bundles, target_label="contrast_dim_level", split_label="game_type",
            position=position,
        )
    except ValueError as e:
        print(f"[probe] cross_game_context skipped: {e}")

    if W_U is not None and a_token_id is not None and b_token_id is not None:
        print("[probe] running logit attribution...", flush=True)
        results["logit_attr"] = logit_attribution(
            bundles, W_U, a_token_id, b_token_id, position=position,
        )

    print("[probe] running subspace decomposition (PD vs Harmony × male vs female)...", flush=True)
    try:
        results["subspace"] = subspace_decomposition(
            bundles,
            label_a="game_type",          class_a1="prisoners_dilemma", class_a2="harmony",
            label_b="contrast_dim_level", class_b1="male",               class_b2="female",
            position=position,
        )
    except ValueError as e:
        print(f"[probe] subspace skipped: {e}")

    return results
