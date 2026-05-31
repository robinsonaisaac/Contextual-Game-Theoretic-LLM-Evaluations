"""Scoring for capability-regression benchmarks (GSM8k math, HumanEval coding).

Unlike the moral/MCQA benchmarks (which score a single ``<decision>A-D</decision>``
letter via :mod:`game_theory_llm.decision_parser`), these need richer scoring:

* **GSM8k** — extract the model's final numeric answer and compare to the gold
  number (exact match after normalising commas / ``$`` / trailing ``.0``).
* **HumanEval** — extract the generated Python function and run it against the
  benchmark's unit tests in an isolated subprocess with a hard timeout
  (execution-based pass@1).

Both operate on the raw generation ``trace`` already stored by the steering
eval shards, so no GPU-side change is needed — we re-score traces locally.

SAFETY: ``humaneval_passes`` executes model-generated code. It does so in a
separate ``python3 -c`` subprocess with a wall-clock timeout, a clean cwd, and
no shell, which is the standard HumanEval harness approach. The candidates are
canonical coding-problem solutions from a small model; still, only run this on
trusted benchmark data.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# GSM8k numeric scoring
# --------------------------------------------------------------------------- #

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
# a signed number, optionally with thousands separators and a decimal part
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _normalize_number(s: str):
    """Parse a loose numeric string to a float, or None. Strips $, commas, %, spaces."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_gsm8k_answer(trace: str):
    """Extract the model's final numeric answer from a GSM8k generation.

    Priority: ``<answer>..</answer>`` tag, then ``\\boxed{..}``, then the last
    number appearing in the text. Returns a float or None.
    """
    if not trace:
        return None
    m = _ANSWER_TAG_RE.search(trace)
    if m:
        v = _normalize_number(m.group(1))
        if v is not None:
            return v
    m = _BOXED_RE.search(trace)
    if m:
        v = _normalize_number(m.group(1))
        if v is not None:
            return v
    nums = _NUMBER_RE.findall(trace)
    if nums:
        return _normalize_number(nums[-1])
    return None


def gold_gsm8k_answer(answer_field: str):
    """Gold number from a GSM8k ``answer`` field (text ending in ``#### N``)."""
    if answer_field is None:
        return None
    tail = answer_field.split("####")[-1]
    return _normalize_number(tail)


def gsm8k_correct(trace: str, gold, tol: float = 1e-4) -> bool:
    """True iff the model's extracted answer matches the gold number."""
    pred = extract_gsm8k_answer(trace)
    g = gold if isinstance(gold, (int, float)) else _normalize_number(str(gold))
    if pred is None or g is None:
        return False
    return abs(pred - g) <= tol * max(1.0, abs(g))


# --------------------------------------------------------------------------- #
# HumanEval code extraction + execution
# --------------------------------------------------------------------------- #

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _trim_blank_lines(s: str) -> str:
    """Drop leading/trailing blank lines but PRESERVE indentation of code
    lines (a bare function-body continuation must keep its leading spaces)."""
    lines = s.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_code(trace: str) -> str:
    """Extract a Python code body from a generation.

    Prefers the last fenced ```python``` block; falls back to the whole trace.
    Indentation is preserved so a bare body (no ``def`` line) still assembles
    correctly when the prompt stub is prepended.
    """
    if not trace:
        return ""
    blocks = _CODE_BLOCK_RE.findall(trace)
    if blocks:
        return _trim_blank_lines(blocks[-1])
    return _trim_blank_lines(trace)


def _build_program(code: str, prompt: str, test: str, entry_point: str) -> str:
    """Assemble an executable program: ensure the entry-point function is
    defined (prepend the HumanEval prompt stub if the model returned only a
    continuation), then append the test harness and the check() call.
    """
    has_def = re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code) is not None
    body = code if has_def else (prompt + "\n" + code)
    return f"{body}\n\n{test}\n\ncheck({entry_point})\n"


def humaneval_passes(trace: str, prompt: str, test: str, entry_point: str,
                     timeout: float = 12.0) -> bool:
    """Execution-based pass: run extracted code + unit tests in a subprocess.

    Returns True iff the program exits 0 (all asserts in ``check`` pass) within
    the timeout. Any exception, timeout, or non-zero exit counts as a failure.
    """
    code = extract_code(trace)
    if not code:
        return False
    program = _build_program(code, prompt, test, entry_point)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "candidate.py"
        src.write_text(program)
        try:
            r = subprocess.run(
                [sys.executable, str(src)],
                cwd=td, capture_output=True, timeout=timeout,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
