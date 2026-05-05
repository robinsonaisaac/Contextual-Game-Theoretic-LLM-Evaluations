"""Serialization for activations, vector sets, and eval results.

Activations are stored one .pt file per story (resumable, easy to inspect).
A Parquet index records which file holds which story so fitting code can
iterate without loading everything into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .models import (
    ActivationBundle,
    SteeringEvalResult,
    SteeringVectorSet,
)


# ---------- ActivationBundle ----------

def save_activation_bundle(bundle: ActivationBundle, dest_dir: Path) -> Path:
    """Write one ActivationBundle to {dest_dir}/{story_id}.pt and return the path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{bundle.story_id}.pt"
    torch.save(
        {
            "story_id": bundle.story_id,
            "model_name": bundle.model_name,
            "decision": bundle.decision,
            "cooperated": bundle.cooperated,
            "prompt_text": bundle.prompt_text,
            "trace_text": bundle.trace_text,
            "activations": bundle.activations,
            "metadata": bundle.metadata,
        },
        path,
    )
    return path


def load_activation_bundle(path: Path) -> ActivationBundle:
    """Load one ActivationBundle from a .pt file."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return ActivationBundle(**blob)


# ---------- Index Parquet ----------

def write_index(bundles: Iterable[ActivationBundle], paths: Iterable[Path],
                index_path: Path, split: str) -> None:
    """Write/append a Parquet index with one row per (story, split)."""
    rows = [
        {
            "story_id": b.story_id,
            "decision": b.decision,
            "cooperated": b.cooperated,
            "split": split,
            "path": str(p),
            "model_name": b.model_name,
        }
        for b, p in zip(bundles, paths)
    ]
    new_df = pd.DataFrame(rows)
    if index_path.exists():
        existing = pd.read_parquet(index_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["story_id", "split"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(index_path, index=False)


def read_index(index_path: Path, split: str | None = None) -> pd.DataFrame:
    """Read the index Parquet, optionally filtered by split."""
    df = pd.read_parquet(index_path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


# ---------- Corpus hashing ----------

def hash_corpus(rows: pd.DataFrame) -> str:
    """Stable hash over (story_id, cooperated) tuples; used as corpus_hash."""
    pairs = sorted(
        (str(r.story_id), bool(r.cooperated)) for r in rows.itertuples(index=False)
    )
    h = hashlib.sha256()
    for sid, coop in pairs:
        h.update(sid.encode())
        h.update(b"\x01" if coop else b"\x00")
    return h.hexdigest()[:16]


# ---------- SteeringVectorSet ----------

def save_vector_set(vs: SteeringVectorSet, path: Path) -> None:
    """Save a SteeringVectorSet to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": vs.model_name,
            "vectors": {
                f"{k[0]}__{k[1]}": {
                    "model_name": v.model_name,
                    "layer": v.layer,
                    "position": v.position,
                    "direction": v.direction,
                    "raw_norm": v.raw_norm,
                    "n_coop": v.n_coop,
                    "n_defect": v.n_defect,
                    "fit_metadata": v.fit_metadata,
                }
                for k, v in vs.vectors.items()
            },
            "corpus_hash": vs.corpus_hash,
            "fit_timestamp": vs.fit_timestamp,
        },
        path,
    )


def load_vector_set(path: Path) -> SteeringVectorSet:
    """Load a SteeringVectorSet."""
    from .models import SteeringVector  # avoid circular import at module level
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vectors = {}
    for k, v in blob["vectors"].items():
        layer_str, position = k.split("__", 1)
        vectors[(int(layer_str), position)] = SteeringVector(**v)
    return SteeringVectorSet(
        model_name=blob["model_name"],
        vectors=vectors,
        corpus_hash=blob["corpus_hash"],
        fit_timestamp=blob["fit_timestamp"],
    )


# ---------- Eval results ----------

def save_eval_results(results: list[SteeringEvalResult], path: Path) -> None:
    """Flatten eval results to a Parquet with one row per (cell, story)."""
    rows = []
    for r in results:
        for d in r.decisions:
            rows.append({
                "layer": r.layer,
                "position": r.position,
                "alpha": r.alpha,
                "story_id": d["story_id"],
                "decision": d["decision"],
                "cooperated": d["cooperated"],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
