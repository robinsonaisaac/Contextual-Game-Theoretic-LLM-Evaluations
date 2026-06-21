"""Local client over the Modal SaemapWorker (safety app, A100-80GB).

Replaces the original local MPS model.py: the heavy 9B forward passes run
remotely; this module marshals prompts out and numpy back. The SAE math stays
local (callers encode the returned residuals with QwenScopeSAE themselves).
"""
from __future__ import annotations
import numpy as np
import modal

APP = "safety"
CLS = "SaemapWorker"
MODEL_NAME = "saemap_9b"

_handle = None

def _worker():
    global _handle
    if _handle is None:
        Cls = modal.Cls.from_name(APP, CLS)          # mirrors run_full_pipeline.py:125
        _handle = Cls(model_name=MODEL_NAME)
    return _handle

def extract_residuals(prompts, layers, completion=None) -> np.ndarray:
    """[N, len(layers), D_MODEL] float32 residuals (mean-pooled over completion
    span if given, else over all prompt tokens)."""
    out = _worker().extract_residuals.remote(list(prompts), list(layers), completion)
    return np.asarray(out, dtype=np.float32)

def pcoop(prompts, coop_letters, fewshot="") -> np.ndarray:
    """[N] normalized P(coop) over {A,B}."""
    out = _worker().pcoop.remote(list(prompts), list(coop_letters), fewshot)
    return np.asarray(out, dtype=np.float32)

def causal_pcoop(prompts, coop_letters, layer, vec, fewshot="") -> np.ndarray:
    """[N] P(coop) with `vec` (4096-d) added at self.layers[layer]. `vec` may be a
    numpy array, torch tensor, or list — coerced to a plain float list for transport."""
    vec_list = np.asarray(vec, dtype=np.float32).reshape(-1).tolist()
    out = _worker().causal_pcoop.remote(
        list(prompts), list(coop_letters), int(layer), vec_list, fewshot)
    return np.asarray(out, dtype=np.float32)
