"""Tier-5 ledger protocol: skeleton IR + canonical render/parse + skeleton verifier.

PURE STDLIB ONLY. Must import under Python 3.9 (generators/tests) AND be importable as a
standalone top-level module inside the Tinker 3.11 venv WITHOUT triggering the heavy
game_theory_llm package __init__ (the RL env does
`sys.path.insert(0, ".../game_theory_llm/reasoning"); import ledger_protocol`).
Therefore: do NOT add any `from game_theory_llm...` import to this file.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Note:
    key: str
    value: str
    deps: List[str] = field(default_factory=list)


@dataclass
class Correction:
    key: str
    wrong_value: str
    right_value: str
    deps: List[str] = field(default_factory=list)


@dataclass
class Restate:
    pass


@dataclass
class Answer:
    value: str


Event = Union[Note, Correction, Restate, Answer]


@dataclass
class LedgerTask:
    family: str
    task_id: str
    prompt: str
    gold_answer: str
    gold_facts: Dict[str, str]
    events: List[Event]
    horizon: int
    needs_ledger: bool = True


_NOTE_RE = re.compile(r"^\s*NOTE:\s*(\S+)\s*=\s*(.+?)\s*$")
_ANS_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$")
_WAS_RE = re.compile(r"\s*\(was\b.*?\)\s*$")


def _clean_value(raw: str) -> str:
    v = raw.split(" <- ")[0]
    v = _WAS_RE.sub("", v)
    return v.strip()


def render_canonical(task: LedgerTask, checkpoint_every: int = 10) -> str:
    """Render the skeleton in the canonical LEDGER/NEXT/NOTE/CHECKPOINT/ANSWER grammar.

    Note       -> `NEXT: derive <key> from <deps>` + `NOTE: <key> = <value> <- <deps>`
    Correction -> `NEXT: recheck <key>`            + `NOTE: <key> = <right> (was <wrong>) <- <deps>`
    Restate    -> `CHECKPOINT: k1=v1; k2=v2; ...` (all facts known so far, in first-seen order)
    Answer     -> `ANSWER: <value>`

    If the skeleton has NO Restate events, a CHECKPOINT is auto-inserted every
    `checkpoint_every` NOTE steps (invariant-4 fallback for short skeletons). CHECKPOINT
    lines are not parsed as claims, so auto-insertion never affects scoring.
    """
    has_restate = any(isinstance(e, Restate) for e in task.events)
    lines = ["LEDGER"]
    known: List[Tuple[str, str]] = []
    known_idx: Dict[str, int] = {}
    steps = 0

    def _set(k: str, v: str) -> None:
        if k in known_idx:
            known[known_idx[k]] = (k, v)
        else:
            known_idx[k] = len(known)
            known.append((k, v))

    for ev in task.events:
        if isinstance(ev, Note):
            frm = (" from " + ", ".join(ev.deps)) if ev.deps else ""
            dep = (" <- " + ", ".join(ev.deps)) if ev.deps else ""
            lines.append(f"NEXT: derive {ev.key}{frm}")
            lines.append(f"NOTE: {ev.key} = {ev.value}{dep}")
            _set(ev.key, ev.value)
            steps += 1
            if not has_restate and checkpoint_every and steps % checkpoint_every == 0:
                lines.append("CHECKPOINT: " + "; ".join(f"{k}={v}" for k, v in known))
        elif isinstance(ev, Correction):
            dep = (" <- " + ", ".join(ev.deps)) if ev.deps else ""
            lines.append(f"NEXT: recheck {ev.key}")
            lines.append(f"NOTE: {ev.key} = {ev.right_value} (was {ev.wrong_value}){dep}")
            _set(ev.key, ev.right_value)
            steps += 1
        elif isinstance(ev, Restate):
            lines.append("CHECKPOINT: " + "; ".join(f"{k}={v}" for k, v in known))
        elif isinstance(ev, Answer):
            lines.append(f"ANSWER: {ev.value}")
    return "\n".join(lines)


def parse_canonical(text: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Extract claimed (key,value) NOTE pairs (Corrections yield right_value) and the answer.
    Order-preserving; CHECKPOINT/NEXT lines are ignored."""
    notes: List[Tuple[str, str]] = []
    answer: Optional[str] = None
    for line in (text or "").splitlines():
        m = _NOTE_RE.match(line)
        if m:
            notes.append((m.group(1).strip(), _clean_value(m.group(2))))
            continue
        a = _ANS_RE.match(line)
        if a:
            answer = a.group(1).strip()
    return notes, answer


def skeleton_claims(events: List[Event]) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Gold extraction straight from the skeleton (no rendering)."""
    notes: List[Tuple[str, str]] = []
    answer: Optional[str] = None
    for ev in events:
        if isinstance(ev, Note):
            notes.append((ev.key, ev.value))
        elif isinstance(ev, Correction):
            notes.append((ev.key, ev.right_value))
        elif isinstance(ev, Answer):
            answer = ev.value
    return notes, answer


def score_claims(notes, answer, gold_facts, gold_answer) -> dict:
    """precision/recall/f1 of claimed notes vs gold key->value; ledger_reward = P*R.

    precision counts EVERY claim (spraying junk notes collapses precision);
    recall counts distinct gold keys stated correctly at least once."""
    claims = [(str(k).strip(), str(v).strip()) for k, v in notes]
    gold = {str(k).strip(): str(v).strip() for k, v in gold_facts.items()}
    n_claims = len(claims)
    correct = sum(1 for k, v in claims if k in gold and gold[k] == v)
    hit_keys = {k for k, v in claims if k in gold and gold[k] == v}
    precision = correct / n_claims if n_claims else 0.0
    recall = len(hit_keys) / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    ans_ok = answer is not None and str(answer).strip() == str(gold_answer).strip()
    return {"precision": precision, "recall": recall, "f1": f1,
            "answer_correct": ans_ok, "ledger_reward": precision * recall}


def _perturb(value: str, rng: random.Random) -> str:
    try:
        n = int(value)
        return str(n + rng.choice([-3, -2, -1, 1, 2, 3]))
    except ValueError:
        cand = value + "_x"
        return cand


def inject_recovery(events, rng) -> List[Event]:
    """Corrupt one Note's value, then insert Restate()+Correction shortly after so the trace
    teaches externalized error-recovery. gold_facts (the correct map) are unchanged."""
    idxs = [i for i, e in enumerate(events) if isinstance(e, Note)]
    if not idxs:
        return list(events)
    i = rng.choice(idxs[:-1]) if len(idxs) > 1 else idxs[0]
    victim = events[i]
    wrong = _perturb(victim.value, rng)
    if wrong == victim.value:
        return list(events)
    out: List[Event] = list(events)
    out[i] = Note(key=victim.key, value=wrong, deps=list(victim.deps))
    ans_pos = next((j for j, e in enumerate(out) if isinstance(e, Answer)), len(out))
    lo = i + 1
    hi = min(ans_pos, i + 4)
    pos = rng.randint(lo, hi) if hi >= lo else ans_pos
    corr = Correction(key=victim.key, wrong_value=wrong,
                      right_value=victim.value, deps=list(victim.deps))
    out[pos:pos] = [Restate(), corr]
    return out
