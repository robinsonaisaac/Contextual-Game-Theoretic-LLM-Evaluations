"""Local client over the Modal SaemapWorker (safety app, A100-80GB).

Replaces the original local MPS model.py: the heavy 9B forward passes run
remotely; this module marshals prompts out and numpy back. The SAE math stays
local (callers encode the returned residuals with QwenScopeSAE themselves).
"""
from __future__ import annotations
import time
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


def _call_with_retry(thunk, label: str, max_attempts: int = 4):
    """Call thunk() up to max_attempts times with exponential backoff.

    Retries on modal.exception.ConnectionError, TimeoutError, or any exception
    whose str representation contains "Deadline" or "deadline".
    Sleeps 5 → 15 → 45 seconds between attempts.
    Re-raises the last exception if all attempts fail.
    """
    sleeps = [5, 15, 45]
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return thunk()
        except Exception as exc:
            exc_str = str(exc)
            is_retryable = (
                isinstance(exc, (TimeoutError,))
                or "deadline" in exc_str.lower()
                or "expired" in exc_str.lower()
                or "conflict" in exc_str.lower()
                or (hasattr(modal, "exception") and isinstance(exc, modal.exception.ConnectionError))
                or type(exc).__name__ in ("ConnectionError", "ConflictError")
            )
            if not is_retryable:
                raise
            last_exc = exc
            if attempt < max_attempts:
                wait = sleeps[attempt - 1]
                print(f"[remote] retry {label} attempt {attempt} after error: {exc_str[:120]}  (sleeping {wait}s)")
                time.sleep(wait)
            else:
                print(f"[remote] retry {label} attempt {attempt} failed (giving up): {exc_str[:120]}")
    raise last_exc


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 6


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

    Splits prompts into chunks of _CHUNK_SIZE and retries each chunk
    independently on transient Modal/gRPC errors.

    Returns a plain list of strings (one per prompt).

    stop_string: if set (e.g. "</decision>"), the returned text is truncated
    at the first occurrence of that string (inclusive).  Use this to keep
    Qwen3.5-9B-Base's long <think> chains within budget.
    """
    prompts = list(prompts)
    results = []
    for chunk_start in range(0, len(prompts), _CHUNK_SIZE):
        chunk = prompts[chunk_start:chunk_start + _CHUNK_SIZE]
        label = f"generate[{chunk_start}:{chunk_start+len(chunk)}]"
        chunk_out = _call_with_retry(
            lambda c=chunk: list(_worker().generate.remote(
                c, int(max_new_tokens), float(temperature), int(seed), stop_string
            )),
            label=label,
        )
        results.extend(chunk_out)
    return results


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

    Splits prompts into chunks of _CHUNK_SIZE and retries each chunk
    independently on transient Modal/gRPC errors (ConnectionError, TimeoutError,
    "Deadline exceeded").

    Returns a plain list of strings (one per prompt).
    """
    vec_list = np.asarray(vec, dtype=np.float32).reshape(-1).tolist()
    prompts = list(prompts)
    results = []
    for chunk_start in range(0, len(prompts), _CHUNK_SIZE):
        chunk = prompts[chunk_start:chunk_start + _CHUNK_SIZE]
        label = f"causal_generate[{chunk_start}:{chunk_start+len(chunk)}] layer={layer}"
        chunk_out = _call_with_retry(
            lambda c=chunk: list(_worker().causal_generate.remote(
                c, int(layer), vec_list,
                int(max_new_tokens), float(temperature), int(seed), stop_string
            )),
            label=label,
        )
        results.extend(chunk_out)
    return results
