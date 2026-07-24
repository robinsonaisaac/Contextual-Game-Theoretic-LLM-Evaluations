"""Staged pipeline to replicate the dilemma-RECOGNITION feature on Qwen3.5-27B.

Stages
------
  extract   -- spawn detached volume extraction job (ONLY this stage is auto-run).
  collect   -- block until job done, pull .pt files from volume, encode with SAE.
  discover  -- diff-of-means + L1 probe per layer; pick best layer by AUC.
  interpret -- max-activating examples for top-5 diffmeans features at best layer.

Usage (from worktree root):
  source ../../.env
  .venv-sae/bin/python scripts/saemap_27b.py --stage extract
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_theory_llm.saemap import corpus, discover, paths
from game_theory_llm.saemap.sae import QwenScopeSAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP = "safety"
CLS = "SaemapWorker"
MODEL_NAME = paths.MODEL_NAME_27B          # "saemap_27b"
RUN_TAG = "saemap_27b"
LAYERS = paths.CANDIDATE_LAYERS_27B       # [16, 24, 32, 40, 48]
D_MODEL = paths.D_MODEL_27B               # 5120
D_SAE = paths.D_SAE_27B                   # 81920
TOPK = paths.TOPK_27B                     # 100
SAE_CACHE = paths.SAE_CACHE_27B
RUN_DIR = paths.RUN_DIR_27B


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def _raw_dir() -> Path:
    d = RUN_DIR / "raw"; d.mkdir(parents=True, exist_ok=True); return d

def _layer_dir(L: int) -> Path:
    d = RUN_DIR / f"L{L}"; d.mkdir(parents=True, exist_ok=True); return d


# ---------------------------------------------------------------------------
# Volume helper (mirrors saemap_collect.py)
# ---------------------------------------------------------------------------
def _volume_get(remote_rel: str, local: Path) -> None:
    subprocess.check_call([
        ".venv-sae/bin/python", "-m", "modal", "volume", "get",
        "--force", "safety", remote_rel, str(local)
    ])


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------
def stage_extract() -> None:
    paths.ensure_run_dirs_27b()
    rec = corpus.recognition_sets(n_per_game=120)
    groups = {
        "recog_dilemma":    {"texts": rec["dilemma"],    "completion": None},
        "recog_nondilemma": {"texts": rec["nondilemma"], "completion": None},
    }
    w = modal.Cls.from_name(APP, CLS)(model_name=MODEL_NAME)
    fc = w.extract_groups_to_volume.spawn(groups, LAYERS, RUN_TAG)
    meta = {
        "call_id": fc.object_id,
        "groups": list(groups.keys()),
        "layers": LAYERS,
        "n": {k: len(v["texts"]) for k, v in groups.items()},
    }
    (RUN_DIR / "extract_call.json").write_text(json.dumps(meta, indent=2))
    print(f"EXTRACT SPAWNED {fc.object_id}")
    print(f"n per group: {meta['n']}")


# ---------------------------------------------------------------------------
# Stage: collect
# ---------------------------------------------------------------------------
def stage_collect() -> None:
    paths.ensure_run_dirs_27b()
    meta = json.loads((RUN_DIR / "extract_call.json").read_text())
    call_id = meta["call_id"]
    groups = meta["groups"]
    layers = meta["layers"]

    print(f"Waiting for job {call_id} ...")
    fc = modal.FunctionCall.from_id(call_id)
    summary = fc.get()
    print("Job summary:", summary)

    # Pull raw residual tensors from volume
    raw = _raw_dir()
    for g in groups:
        remote_rel = f"runs/{RUN_TAG}/activations/{g}.pt"
        print(f"  volume get {remote_rel} -> {raw / f'{g}.pt'}")
        _volume_get(remote_rel, raw / f"{g}.pt")

    # SAE-encode per layer
    for li, L in enumerate(layers):
        sae = QwenScopeSAE.load(L, k=TOPK, cache_dir=SAE_CACHE)
        out = _layer_dir(L)
        for g in groups:
            blob = torch.load(raw / f"{g}.pt", map_location="cpu")
            resid = blob["residuals"].float()[:, li, :]     # [N, D_MODEL]
            feats = sae.encode(resid)                        # [N, D_SAE]
            torch.save(feats, out / f"{g}.pt")
        print(f"  L{L} encoded {len(groups)} groups  shape={feats.shape}")

    print("COLLECT DONE")


# ---------------------------------------------------------------------------
# Stage: discover
# ---------------------------------------------------------------------------
def stage_discover() -> None:
    paths.ensure_run_dirs_27b()
    best_layer = None
    best_auc = -1.0

    for L in LAYERS:
        pos = torch.load(_layer_dir(L) / "recog_dilemma.pt",    map_location="cpu").numpy()
        neg = torch.load(_layer_dir(L) / "recog_nondilemma.pt", map_location="cpu").numpy()

        X = np.concatenate([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

        dom = discover.diff_of_means(pos, neg)
        probe = discover.l1_probe(X, y, C=0.05)

        coef_arr = np.array(probe["coef"])
        result = {
            "layer": L,
            "auc": probe["auc"],
            "n_nonzero": probe["n_nonzero"],
            "diffmeans_top": discover.top_features(dom, 10),
            "probe_top": [int(i) for i in np.argsort(-np.abs(coef_arr))[:10]],
        }
        (RUN_DIR / f"discover_recog_fine_L{L}.json").write_text(json.dumps(result, indent=2))
        print(f"L{L}  auc={result['auc']:.3f}  n_nz={result['n_nonzero']}")

        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_layer = L

    summary = {"best_layer": best_layer, "best_auc": best_auc}
    (RUN_DIR / "discover_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best layer: L{best_layer}  auc={best_auc:.3f}")


# ---------------------------------------------------------------------------
# Stage: interpret
# ---------------------------------------------------------------------------
def stage_interpret() -> None:
    paths.ensure_run_dirs_27b()

    # Load best layer from discover summary
    summary_path = RUN_DIR / "discover_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run --stage discover first: {summary_path}")
    summary = json.loads(summary_path.read_text())
    best_layer = summary["best_layer"]

    # Top-5 diff-of-means features at best layer
    disc_path = RUN_DIR / f"discover_recog_fine_L{best_layer}.json"
    disc = json.loads(disc_path.read_text())
    top5_features = disc["diffmeans_top"][:5]
    print(f"Best layer: L{best_layer}  top-5 diffmeans features: {top5_features}")

    # Build interpretation pool
    pool = (
        corpus.game_texts(paths.DILEMMA_GAMES + paths.NONDILEMMA_GAMES, 40)
        + corpus.nongame_texts(120)
    )
    print(f"Interp pool size: {len(pool)}")

    # Call 27B worker directly (remote.extract_residuals is hardwired to 9B)
    print(f"Fetching residuals from 27B worker at L{best_layer} ...")
    w = modal.Cls.from_name(APP, CLS)(model_name=MODEL_NAME)
    raw_resid = w.extract_residuals.remote(list(pool), [best_layer], None)
    resid_np = np.asarray(raw_resid, dtype=np.float32)   # [N, 1, D_MODEL]
    resid_t = torch.tensor(resid_np[:, 0, :], dtype=torch.float32)  # [N, D_MODEL]

    # SAE encode
    sae = QwenScopeSAE.load(best_layer, k=TOPK, cache_dir=SAE_CACHE)
    acts = sae.encode(resid_t)   # [N, D_SAE]

    # Max-activating examples per feature
    TOPN = 8
    features_out: dict[str, list[dict]] = {}
    for f in top5_features:
        col = acts[:, f]
        k = min(TOPN, len(pool))
        order = torch.argsort(col, descending=True)[:k].tolist()
        snippets = [{"score": float(col[i].item()), "snippet": pool[i][:300]} for i in order]
        features_out[str(f)] = snippets
        print(f"\nFeature {f}  (top score={snippets[0]['score']:.4f}):")
        for s in snippets[:3]:
            print(f"  [{s['score']:.3f}] {s['snippet'][:120]!r}")

    result = {
        "best_layer": best_layer,
        "top5_features": top5_features,
        "examples": features_out,
    }
    (RUN_DIR / "interpret.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {RUN_DIR / 'interpret.json'}")
    print("INTERPRET DONE")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="27B recognition SAE pipeline")
    ap.add_argument("--stage", required=True,
                    choices=["extract", "collect", "discover", "interpret"])
    args = ap.parse_args()

    if args.stage == "extract":
        stage_extract()
    elif args.stage == "collect":
        stage_collect()
    elif args.stage == "discover":
        stage_discover()
    elif args.stage == "interpret":
        stage_interpret()


if __name__ == "__main__":
    main()
