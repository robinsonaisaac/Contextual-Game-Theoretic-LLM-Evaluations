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

def ab_mass(prompts, fewshot="") -> np.ndarray:
    """[N] fraction of next-token mass on {A,B} after '\\n<decision>' (Step-0 gate)."""
    out = _worker().ab_mass.remote(list(prompts), fewshot)
    return np.asarray(out, dtype=np.float32)

def generate(prompts, max_new_tokens=1500, temperature=0.0, seed=0,
             stop_string=None) -> list:
    """Generate free-text continuations for each prompt via the Modal worker.

    Returns a plain list of strings (one per prompt).

    stop_string: if set (e.g. "</decision>"), the returned text is truncated
    at the first occurrence of that string (inclusive).  Use this to keep
    Qwen3.5-9B-Base's long <think> chains within budget.
    """
    return list(_worker().generate.remote(
        list(prompts), int(max_new_tokens), float(temperature), int(seed),
        stop_string))

def causal_pcoop(prompts, coop_letters, layer, vec, fewshot="") -> np.ndarray:
    """[N] P(coop) with `vec` (4096-d) added at self.layers[layer]. `vec` may be a
    numpy array, torch tensor, or list — coerced to a plain float list for transport."""
    vec_list = np.asarray(vec, dtype=np.float32).reshape(-1).tolist()
    out = _worker().causal_pcoop.remote(
        list(prompts), list(coop_letters), int(layer), vec_list, fewshot)
    return np.asarray(out, dtype=np.float32)

def causal_generate(prompts, layer, vec, max_new_tokens=2000, temperature=0.0,
                    seed=0, stop_string=None) -> list:
    """Generate continuations with `vec` (4096-d) injected at self.layers[layer] block input.

    Injection locus matches extract_residuals / causal_pcoop (resid_pre[L]).
    `vec` may be a numpy array, torch tensor, or list — coerced to float list for transport.

    Returns a plain list of strings (one per prompt).
    """
    vec_list = np.asarray(vec, dtype=np.float32).reshape(-1).tolist()
    return list(_worker().causal_generate.remote(
        list(prompts), int(layer), vec_list,
        int(max_new_tokens), float(temperature), int(seed), stop_string))
