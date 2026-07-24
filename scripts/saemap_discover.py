"""Run discovery (A diff-of-means, B L1 probe, C decomposition) per contrast/layer.

Cache layout expected (written by Task 6 / saemap_collect.py):
  SAE features: data/runs/saemap_9b/activations/L{L}/{group}.pt
    -> tensor [N, D_SAE] (float32)
  Raw residuals: data/runs/saemap_9b/activations/raw/{group}.pt
    -> dict {"residuals": tensor[N, len(layers), D_MODEL] (bf16),
             "layers": [...], "n": N}

Groups: decision_coop, decision_defect,
        recog_dilemma, recog_nondilemma,
        recog_game, recog_nongame
Layers: paths.CANDIDATE_LAYERS = [8, 12, 16, 20, 24]
"""
import json
import numpy as np
import torch
from game_theory_llm.saemap import discover, paths
from game_theory_llm.saemap.sae import QwenScopeSAE


def _load_sae(layer: int, group: str) -> np.ndarray:
    """Load [N, D_SAE] SAE feature tensor for a group at a given layer."""
    p = paths.RUN_DIR / "activations" / f"L{layer}" / f"{group}.pt"
    return torch.load(p, map_location="cpu").numpy()


def _load_raw(group: str) -> dict:
    """Load raw residual dict for a group (layer-independent file)."""
    p = paths.RUN_DIR / "activations" / "raw" / f"{group}.pt"
    return torch.load(p, map_location="cpu")


def _contrast(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Run passes A (diff-of-means) and B (L1 probe) for a pos/neg pair."""
    X = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    dom = discover.diff_of_means(pos, neg)
    probe = discover.l1_probe(X, y, C=0.05)
    coef_arr = np.array(probe["coef"])
    return {
        "auc": probe["auc"],
        "n_nonzero": probe["n_nonzero"],
        "diffmeans_top": discover.top_features(dom, 10),
        "probe_top": [int(i) for i in np.argsort(-np.abs(coef_arr))[:10]],
    }


def main() -> None:
    paths.ensure_run_dirs()

    contrasts = {
        "decision": ("decision_coop", "decision_defect"),
        "recog_fine": ("recog_dilemma", "recog_nondilemma"),
        "recog_coarse": ("recog_game", "recog_nongame"),
    }

    # Pre-load raw residuals for decision pass C (layer-independent file, one load)
    decision_raw_coop = None
    decision_raw_defect = None

    summary = {}
    for contrast_name, (pos_group, neg_group) in contrasts.items():
        best = None

        for L in paths.CANDIDATE_LAYERS:
            pos = _load_sae(L, pos_group)
            neg = _load_sae(L, neg_group)

            res = _contrast(pos, neg)
            res["layer"] = L

            # Pass C: holistic steering-direction decomposition (decision contrast only)
            if contrast_name == "decision":
                # Lazy-load raw residuals once (same file for all layers)
                if decision_raw_coop is None:
                    decision_raw_coop = _load_raw("decision_coop")
                    decision_raw_defect = _load_raw("decision_defect")

                layer_idx = decision_raw_coop["layers"].index(L)
                # residuals: [N, len(layers), D_MODEL] in bf16 -> float32 -> numpy
                coop_resid = decision_raw_coop["residuals"][:, layer_idx, :].float().numpy()
                defect_resid = decision_raw_defect["residuals"][:, layer_idx, :].float().numpy()
                direction = coop_resid.mean(0) - defect_resid.mean(0)  # [D_MODEL]

                W_dec = QwenScopeSAE.load(L).W_dec.numpy()  # [D_MODEL, D_SAE]
                decomp = discover.decompose_direction(direction, W_dec, topn=10)
                res["decomp_top"] = decomp
                res["decomp_overlap"] = len(
                    set(i for i, _ in decomp) & set(res["probe_top"])
                )

            out_path = paths.RUN_DIR / "discover" / f"{contrast_name}_L{L}.json"
            out_path.write_text(json.dumps(res, indent=2))

            if best is None or res["auc"] > best["auc"]:
                best = res

        summary[contrast_name] = {
            "best_layer": best["layer"],
            "best_auc": best["auc"],
            "n_nonzero": best["n_nonzero"],
        }
        print(f"{contrast_name}: best L{best['layer']} auc={best['auc']:.3f}")

    (paths.RUN_DIR / "discover" / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DISCOVER DONE")


if __name__ == "__main__":
    main()
