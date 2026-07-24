# Tier-5 Domain-General Inline Bookkeeping ("ledger") — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the generators, protocol, dataset, SFT→RL training, and generalization eval battery that test whether an inline state-ledger bookkeeping loop is a *trainable, domain-general technique* that transfers to unseen long-horizon domains (spec `2026-07-02-tier5-inline-ledger-design.md`).

**Architecture:** A single pure-stdlib module `ledger_protocol.py` owns the skeleton IR (Note/Correction/Restate/Answer events), the canonical LEDGER/NEXT/NOTE/CHECKPOINT/ANSWER renderer + paired parser, the skeleton-level verifier `score_claims`, and `inject_recovery`. Six seeded generators (4 train + 2 held-out) emit `LedgerTask` skeletons; a Sonnet naturalization pipeline rephrases ~75% of traces under rotated style seeds with mandatory extract-back round-trip QC. Phase-1 LoRA SFT (Tinker, Qwen3-30B-A3B) installs the behavior; a cheap gate eval decides whether Phase-2 GRPO (dense reward `answer_correct + 0.5·ledger_reward`) hardens it; a final battery measures in-domain extrapolation, held-out families, and 3 real BBH benchmarks + a GSM8K control, with an LLM-judge invariant-usage attribution and a suppression ablation.

**Tech Stack:** Python 3.9 (system `python3`) for pure-python generators/protocol/dataset + `pytest`; Python 3.11 (`.venv-tinker/bin/python`) for all Tinker SFT/GRPO/eval; OpenRouter (`openai` + `python-dotenv`, system `python3`) for Sonnet naturalization/judging; HuggingFace `datasets` for BBH fetch.

## Global Constraints

- **Worktree:** `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering`, branch `feature/activation-steering`, cwd for all commands.
- **Pure-python code + tests:** system `python3` (3.9) — code must be 3.9-compatible (`from __future__ import annotations`); tests `python3 -m pytest tests/... -v`.
- **All Tinker scripts:** `.venv-tinker/bin/python` (3.11). Tinker billing confirmed live 2026-07-02. Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`. Save checkpoints frequently (billing-wall lesson).
- **OpenRouter calls (naturalization/judging):** system `python3` with `dotenv` load of the main-repo .env (`/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env`); judge model `anthropic/claude-sonnet-4-6`; NEVER print API keys.
- **Evals:** temperature 0; final answers parsed mechanically from a required `ANSWER: <value>` tail (accept `<answer>` tags too); quality/invariant judgments ALWAYS via LLM judge, never regex (project rule).
- **Success thresholds (spec §7):** in-domain extrapolated ≥ +15pp; held-out families ≥ +10pp; real benchmarks uplift on ≥2 of 3 with GSM8K degradation ≤ 2pp. Claim discipline: "generalizable technique" only if held-out AND benchmarks pass.

### Pinned shared interfaces (use EXACTLY these across every task — the type-consistency backbone)

```python
# game_theory_llm/reasoning/ledger_protocol.py — PURE STDLIB; no `from game_theory_llm...` import
@dataclass
class Note:       key: str; value: str; deps: List[str]
@dataclass
class Correction: key: str; wrong_value: str; right_value: str; deps: List[str]
@dataclass
class Restate:    pass
@dataclass
class Answer:     value: str
Event = Union[Note, Correction, Restate, Answer]

@dataclass
class LedgerTask:
    family: str; task_id: str; prompt: str
    gold_answer: str
    gold_facts: Dict[str, str]       # key -> final correct value
    events: List[Event]              # gold skeleton, ends with Answer
    horizon: int                     # count of Note+Correction events
    needs_ledger: bool = True

def render_canonical(task: LedgerTask, checkpoint_every: int = 10) -> str
def parse_canonical(text: str) -> Tuple[List[Tuple[str, str]], Optional[str]]
def skeleton_claims(events: List[Event]) -> Tuple[List[Tuple[str, str]], Optional[str]]
def score_claims(notes, answer, gold_facts, gold_answer) -> dict   # precision,recall,f1,answer_correct,ledger_reward=P*R
def inject_recovery(events, rng) -> List[Event]
# game_theory_llm/reasoning/ledger_tasks/__init__.py
def get_generator(family: str) -> Callable[[int, int], LedgerTask]   # each module exposes gen(seed, horizon) -> LedgerTask
```

- **Universal generator test (every family):** replaying `skeleton_claims(task.events)` through `score_claims` yields precision=recall=1.0 and answer_correct=True; `gen(seed, h)` is deterministic per seed.
- **Naturalization QC rule:** claimed `(key,value)` SET equality vs the skeleton's `gold_facts` + answer match (order NOT enforced); reject-and-regenerate otherwise; report pass-rate.
- **Phase-2 reward:** `answer_correct + 0.5 * ledger_reward` via `parse_canonical`+`score_claims`; rollout prompts carry the canonical-format anchor.

---

## Task 1: `ledger_protocol.py` — skeleton IR, canonical render/parse, verifier, recovery

**Files:**
- Create: `game_theory_llm/reasoning/ledger_protocol.py`
- Test: `tests/test_ledger_protocol.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces: `Note`, `Correction`, `Restate`, `Answer`, `Event`, `LedgerTask`, `render_canonical(task, checkpoint_every=10) -> str`, `parse_canonical(text) -> Tuple[List[Tuple[str,str]], Optional[str]]`, `skeleton_claims(events) -> Tuple[List[Tuple[str,str]], Optional[str]]`, `score_claims(notes, answer, gold_facts, gold_answer) -> dict`, `inject_recovery(events, rng) -> List[Event]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ledger_protocol.py
import random
from game_theory_llm.reasoning.ledger_protocol import (
    Answer, Correction, LedgerTask, Note, Restate,
    render_canonical, parse_canonical, skeleton_claims, score_claims, inject_recovery,
)


def _task():
    events = [
        Note("a", "1", []),
        Note("b", "2", ["a"]),
        Note("c", "3", ["a", "b"]),
        Answer("3"),
    ]
    return LedgerTask("demo", "demo_1", "prompt body", "3",
                      {"a": "1", "b": "2", "c": "3"}, events, horizon=3)


def test_gold_skeleton_scores_perfect():
    t = _task()
    notes, ans = skeleton_claims(t.events)
    s = score_claims(notes, ans, t.gold_facts, t.gold_answer)
    assert s["precision"] == 1.0 and s["recall"] == 1.0
    assert s["answer_correct"] is True and s["ledger_reward"] == 1.0


def test_render_parse_roundtrip():
    t = _task()
    notes, ans = parse_canonical(render_canonical(t))
    assert notes == [("a", "1"), ("b", "2"), ("c", "3")]
    assert ans == "3"


def test_render_has_grammar_tokens():
    text = render_canonical(_task())
    assert text.startswith("LEDGER")
    assert "NEXT:" in text and "NOTE:" in text and "ANSWER: 3" in text


def test_spray_guard_collapses_precision():
    gold = {"a": "1", "b": "2", "c": "3"}
    notes = [("a", "1"), ("b", "2"), ("c", "3")] + [(f"j{i}", "9") for i in range(30)]
    s = score_claims(notes, "3", gold, "3")
    assert s["recall"] == 1.0
    assert s["precision"] < 0.2 and s["ledger_reward"] < 0.2


def test_correction_yields_right_value():
    text = "LEDGER\nNOTE: a = 5 (was 4) <- x\nANSWER: 5"
    notes, ans = parse_canonical(text)
    assert notes == [("a", "5")] and ans == "5"


def test_inject_recovery_adds_restate_and_correction():
    t = _task()
    ev = inject_recovery(t.events, random.Random(0))
    assert any(isinstance(e, Restate) for e in ev)
    assert any(isinstance(e, Correction) for e in ev)
    # after recovery the final claimed value for the corrupted key is correct again
    final = {}
    for e in ev:
        if isinstance(e, Note):
            final[e.key] = e.value
        elif isinstance(e, Correction):
            final[e.key] = e.right_value
    assert final == t.gold_facts
    assert isinstance(ev[-1], Answer) and ev[-1].value == "3"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_theory_llm.reasoning.ledger_protocol'`.

- [ ] **Step 3: Write the implementation**

```python
# game_theory_llm/reasoning/ledger_protocol.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_ledger_protocol.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/reasoning/ledger_protocol.py tests/test_ledger_protocol.py
git commit -m "feat(tier5): ledger protocol IR + canonical render/parse + verifier + recovery"
```

---

## Task 2: `ledger_tasks` base registry + `trees.py` generator

**Files:**
- Create: `game_theory_llm/reasoning/ledger_tasks/__init__.py`
- Create: `game_theory_llm/reasoning/ledger_tasks/trees.py`
- Test: `tests/test_ledger_tasks.py`

**Interfaces:**
- Consumes: `game_theory_llm.reasoning.ledger_protocol` (`Note`, `Restate`, `Answer`, `Event`, `LedgerTask`, `skeleton_claims`, `score_claims`, `render_canonical`, `parse_canonical`); `game_theory_llm.reasoning.gametree` (`_gen_tree(rng, depth, branching, leaf_lo, leaf_hi)`, `minimax(node, maximizing)`).
- Produces: `get_generator(family) -> Callable[[int,int], LedgerTask]`; `TRAIN_FAMILIES`, `HELDOUT_FAMILIES`; `trees.gen(seed, horizon) -> LedgerTask` (family `"trees"`; keys `v.<path>`; horizon knob = depth d3–d7 via `round(log2(horizon+1))`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger_tasks.py
import pytest
from game_theory_llm.reasoning.ledger_protocol import (
    Answer, skeleton_claims, score_claims, render_canonical, parse_canonical)
from game_theory_llm.reasoning.ledger_tasks import get_generator

# Grows as families land: Task 2 = trees; Task 3 += register_machine, graph_search;
# Task 4 += forward_chain, object_tracking, scheduling.
IMPLEMENTED = ["trees"]


@pytest.mark.parametrize("family", IMPLEMENTED)
@pytest.mark.parametrize("horizon", [12, 24, 48])
def test_gold_skeleton_scores_perfect(family, horizon):
    task = get_generator(family)(seed=7, horizon=horizon)
    notes, ans = skeleton_claims(task.events)
    s = score_claims(notes, ans, task.gold_facts, task.gold_answer)
    assert s["precision"] == 1.0 and s["recall"] == 1.0
    assert s["answer_correct"] is True
    assert isinstance(task.events[-1], Answer)
    assert task.horizon == len(task.gold_facts)


@pytest.mark.parametrize("family", IMPLEMENTED)
def test_deterministic(family):
    a = get_generator(family)(seed=3, horizon=30)
    b = get_generator(family)(seed=3, horizon=30)
    assert a.gold_answer == b.gold_answer and a.gold_facts == b.gold_facts
    assert render_canonical(a) == render_canonical(b)


@pytest.mark.parametrize("family", IMPLEMENTED)
def test_canonical_roundtrip(family):
    task = get_generator(family)(seed=11, horizon=20)
    assert parse_canonical(render_canonical(task)) == skeleton_claims(task.events)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_theory_llm.reasoning.ledger_tasks'`.

- [ ] **Step 3: Write the base registry**

```python
# game_theory_llm/reasoning/ledger_tasks/__init__.py
"""Registry of Tier-5 ledger-task generators. Each family module exposes
`gen(seed: int, horizon: int) -> LedgerTask`. get_generator imports lazily so a
partially-implemented package still imports (families land across Tasks 2-4)."""
from __future__ import annotations

import importlib
from typing import Callable

from game_theory_llm.reasoning.ledger_protocol import LedgerTask

TRAIN_FAMILIES = ["trees", "register_machine", "graph_search", "forward_chain"]
HELDOUT_FAMILIES = ["object_tracking", "scheduling"]


def get_generator(family: str) -> "Callable[[int, int], LedgerTask]":
    mod = importlib.import_module(f"game_theory_llm.reasoning.ledger_tasks.{family}")
    return mod.gen
```

- [ ] **Step 4: Write `trees.py`**

```python
# game_theory_llm/reasoning/ledger_tasks/trees.py
"""Minimax game-tree family. Wraps gametree._gen_tree/minimax. Ledger content: each
internal node's backward-induction value keyed by its path id (`v.<path>`); deps = the
child node keys. horizon knob = depth d3-d7 chosen from the requested horizon."""
from __future__ import annotations

import math
import random
from typing import List

from game_theory_llm.reasoning.gametree import _gen_tree
from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10


def _depth_for_horizon(horizon: int) -> int:
    d = int(round(math.log2(max(2, horizon) + 1)))
    return max(3, min(7, d))


def _render_leaves(node, path, maximizing, lines):
    who = "MAX" if maximizing else "MIN"
    for lab, c in node.children:
        p = f"{path}{lab}"
        if c.is_leaf:
            lines.append(f"- {who} plays {lab} -> outcome {c.value:+d}  [{p}]")
        else:
            lines.append(f"- {who} plays {lab}:  [{p}]")
            _render_leaves(c, p, not maximizing, lines)


def _walk(node, path, maximizing, notes):
    if node.is_leaf:
        return node.value, f"leaf.{path}"
    child = []
    for lab, c in node.children:
        v, k = _walk(c, f"{path}{lab}", not maximizing, notes)
        child.append((v, k))
    val = (max if maximizing else min)(cv for cv, _ in child)
    key = f"v.{path}"
    notes.append(Note(key=key, value=str(val), deps=[k for _, k in child]))
    return val, key


def gen(seed: int, horizon: int) -> LedgerTask:
    depth = _depth_for_horizon(horizon)
    rng = random.Random(seed)
    root = _gen_tree(rng, depth, branching=2, leaf_lo=-9, leaf_hi=9)
    notes: List[Note] = []
    root_val, _ = _walk(root, "R", True, notes)
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    events.append(Answer(value=str(root_val)))
    gold_facts = {n.key: n.value for n in notes}
    lines: List[str] = []
    _render_leaves(root, "R", True, lines)
    prompt = (
        "Two players MAX and MIN alternate turns; MAX moves first and wants the final "
        "outcome as LARGE as possible, MIN as SMALL as possible; both play optimally.\n"
        f"Game tree ({depth} moves deep); each node is tagged with its path id in [brackets]:\n"
        + "\n".join(lines) + "\n\n"
        "Work bottom-up (backward induction). Keep an explicit running ledger of each "
        "internal node's value keyed by its path id, then read off the root value.\n"
        "End with a line: ANSWER: <the root value under optimal play>."
    )
    return LedgerTask(
        family="trees", task_id=f"trees_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=str(root_val), gold_facts=gold_facts, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: PASS — 4 passed (`trees` × 3 horizons + determinism + roundtrip).

- [ ] **Step 6: Commit**

```bash
git add game_theory_llm/reasoning/ledger_tasks/__init__.py \
        game_theory_llm/reasoning/ledger_tasks/trees.py tests/test_ledger_tasks.py
git commit -m "feat(tier5): ledger-task registry + trees (minimax) generator"
```

---

## Task 3: `register_machine.py` + `graph_search.py` generators

**Files:**
- Create: `game_theory_llm/reasoning/ledger_tasks/register_machine.py`
- Create: `game_theory_llm/reasoning/ledger_tasks/graph_search.py`
- Modify: `tests/test_ledger_tasks.py` (extend `IMPLEMENTED`)

**Interfaces:**
- Consumes: `ledger_protocol` (`Answer`, `Event`, `LedgerTask`, `Note`, `Restate`).
- Produces: `register_machine.gen(seed, horizon)` (family `"register_machine"`; keys `s<step>.r<reg>`; deps = latest keys of operand registers; answer = value of a queried register mod 1000). `graph_search.gen(seed, horizon)` (family `"graph_search"`; keys `dist.<node>`; deps = predecessor dist key; answer = BFS distance from node 0 to the farthest node).

- [ ] **Step 1: Extend the failing test**

Modify `tests/test_ledger_tasks.py` line `IMPLEMENTED = ["trees"]` to:

```python
IMPLEMENTED = ["trees", "register_machine", "graph_search"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_theory_llm.reasoning.ledger_tasks.register_machine'`.

- [ ] **Step 3: Write `register_machine.py`**

```python
# game_theory_llm/reasoning/ledger_tasks/register_machine.py
"""Register-machine simulation. 4 registers, arithmetic mod 1000. Ledger content: each
write keyed `s<step>.r<reg>` = value; deps = the latest keys writing the two operands.
horizon = number of Note events (4 init writes + N program steps)."""
from __future__ import annotations

import random
from typing import List

from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10
OPS = [("+", lambda a, b: (a + b) % 1000),
       ("-", lambda a, b: (a - b) % 1000),
       ("*", lambda a, b: (a * b) % 1000)]


def gen(seed: int, horizon: int) -> LedgerTask:
    rng = random.Random(seed)
    n_reg = 4
    n_steps = max(1, horizon - n_reg)
    notes: List[Note] = []
    latest, value, prog = {}, {}, []
    for i in range(n_reg):
        v = rng.randint(0, 99)
        key = f"s0.r{i}"
        value[i], latest[i] = v, key
        notes.append(Note(key=key, value=str(v), deps=[]))
        prog.append(f"init r{i} = {v}")
    for t in range(1, n_steps + 1):
        dest, a, b = rng.randrange(n_reg), rng.randrange(n_reg), rng.randrange(n_reg)
        sym, fn = rng.choice(OPS)
        res = fn(value[a], value[b])
        key = f"s{t}.r{dest}"
        notes.append(Note(key=key, value=str(res), deps=[latest[a], latest[b]]))
        prog.append(f"step {t}: r{dest} = r{a} {sym} r{b}")
        value[dest], latest[dest] = res, key
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    query = rng.randrange(n_reg)
    ans = str(value[query])
    events.append(Answer(value=ans))
    prompt = (
        "A register machine has 4 registers r0..r3; all arithmetic is mod 1000.\n"
        + "\n".join(prog) + "\n\n"
        "Simulate step by step. Keep an explicit ledger noting each write as "
        "`s<step>.r<reg> = <value>`.\n"
        f"After the program, report the value of r{query}.\n"
        "End with a line: ANSWER: <value of the queried register mod 1000>."
    )
    return LedgerTask(
        family="register_machine", task_id=f"regm_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=ans, gold_facts={n.key: n.value for n in notes}, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 4: Write `graph_search.py`**

```python
# game_theory_llm/reasoning/ledger_tasks/graph_search.py
"""BFS shortest-path family over a random tree (guarantees connectivity + a unique BFS
parent per node). Ledger content: each node's distance keyed `dist.<node>`; deps =
predecessor dist key. horizon = number of nodes. Answer = distance to the farthest node."""
from __future__ import annotations

import random
from collections import deque
from typing import List

from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10


def gen(seed: int, horizon: int) -> LedgerTask:
    rng = random.Random(seed)
    V = max(3, horizon)
    parent = {0: None}
    edges = []
    adj = {i: [] for i in range(V)}
    for v in range(1, V):
        u = rng.randrange(0, v)
        parent[v] = u
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
    dist = {0: 0}
    order = [0]
    q = deque([0])
    while q:
        u = q.popleft()
        for w in sorted(adj[u]):
            if w not in dist:
                dist[w] = dist[u] + 1
                order.append(w)
                q.append(w)
    notes: List[Note] = []
    for v in order:
        if v == 0:
            notes.append(Note(key="dist.0", value="0", deps=[]))
        else:
            notes.append(Note(key=f"dist.{v}", value=str(dist[v]), deps=[f"dist.{parent[v]}"]))
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    target = max(range(V), key=lambda x: (dist[x], -x))
    ans = str(dist[target])
    events.append(Answer(value=ans))
    elist = "; ".join(f"{u}-{v}" for u, v in edges)
    prompt = (
        f"An undirected graph has {V} nodes (0..{V - 1}) and these edges: {elist}.\n"
        "Compute shortest-path distances from node 0 by breadth-first search. Keep an "
        "explicit ledger noting each node's distance as `dist.<node> = <d>`.\n"
        f"Report the distance from node 0 to node {target}.\n"
        "End with a line: ANSWER: <shortest-path distance>."
    )
    return LedgerTask(
        family="graph_search", task_id=f"graph_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=ans, gold_facts={n.key: n.value for n in notes}, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: PASS — 12 passed (3 families × [3 horizons + determinism + roundtrip]).

- [ ] **Step 6: Commit**

```bash
git add game_theory_llm/reasoning/ledger_tasks/register_machine.py \
        game_theory_llm/reasoning/ledger_tasks/graph_search.py tests/test_ledger_tasks.py
git commit -m "feat(tier5): register-machine + graph-search ledger generators"
```

---

## Task 4: `forward_chain.py` + `object_tracking.py` [HELD-OUT] + `scheduling.py` [HELD-OUT]

**Files:**
- Create: `game_theory_llm/reasoning/ledger_tasks/forward_chain.py`
- Create: `game_theory_llm/reasoning/ledger_tasks/object_tracking.py`
- Create: `game_theory_llm/reasoning/ledger_tasks/scheduling.py`
- Modify: `tests/test_ledger_tasks.py` (extend `IMPLEMENTED` to all 6)

**Interfaces:**
- Consumes: `ledger_protocol` (`Answer`, `Event`, `LedgerTask`, `Note`, `Restate`).
- Produces: `forward_chain.gen` (family `"forward_chain"`; keys = fact names `f<i>`; answer = `yes`/`no`). `object_tracking.gen` (family `"object_tracking"`, HELD-OUT; keys `loc.<entity>.<t>`; answer = final room color). `scheduling.gen` (family `"scheduling"`, HELD-OUT; keys `finish.<task>`; answer = project completion time).

- [ ] **Step 1: Extend the failing test**

Modify `tests/test_ledger_tasks.py` line to:

```python
IMPLEMENTED = ["trees", "register_machine", "graph_search",
               "forward_chain", "object_tracking", "scheduling"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_theory_llm.reasoning.ledger_tasks.forward_chain'`.

- [ ] **Step 3: Write `forward_chain.py`**

```python
# game_theory_llm/reasoning/ledger_tasks/forward_chain.py
"""Forward-chaining Horn deduction over a linear implication chain. Ledger content: each
derived fact keyed by its name (`f<i>`) = true; deps = the antecedent fact key. horizon =
number of facts. Answer = whether the query fact is derivable (yes/no)."""
from __future__ import annotations

import random
from typing import List

from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10


def gen(seed: int, horizon: int) -> LedgerTask:
    rng = random.Random(seed)
    L = max(2, horizon)
    notes: List[Note] = [Note(key="f0", value="true", deps=[])]
    rules = []
    for i in range(1, L):
        notes.append(Note(key=f"f{i}", value="true", deps=[f"f{i - 1}"]))
        rules.append(f"if f{i - 1} then f{i}")
    rng.shuffle(rules)
    if rng.random() < 0.5:
        query, ans = f"f{L - 1}", "yes"
    else:
        query, ans = f"g{L}", "no"          # g{L} is never derivable from the chain
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    events.append(Answer(value=ans))
    prompt = (
        "Forward-chaining deduction over Horn rules.\nKnown fact: f0.\nRules:\n"
        + "\n".join(rules) + "\n\n"
        "Derive every fact that follows. Keep an explicit ledger noting each fact as "
        "`<fact> = true` with the rule/facts it came from.\n"
        f"Question: is {query} derivable?\n"
        "End with a line: ANSWER: yes or ANSWER: no."
    )
    return LedgerTask(
        family="forward_chain", task_id=f"chain_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=ans, gold_facts={n.key: n.value for n in notes}, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 4: Write `object_tracking.py` [HELD-OUT]**

```python
# game_theory_llm/reasoning/ledger_tasks/object_tracking.py
"""HELD-OUT family (never enters training). Multi-entity object tracking: entities swap
rooms; ledger content: each entity's room after every swap keyed `loc.<entity>.<t>`;
deps = the two swapped entities' latest keys. Answer = final room of a queried entity."""
from __future__ import annotations

import random
from typing import List

from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10
_ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank"]
_LOCS = ["red", "green", "blue", "yellow", "purple", "orange"]


def gen(seed: int, horizon: int) -> LedgerTask:
    rng = random.Random(seed)
    n_ent = min(len(_ENTITIES), max(3, min(6, horizon // 6 + 3)))
    ents, locs = _ENTITIES[:n_ent], _LOCS[:n_ent]
    order = list(range(n_ent))
    rng.shuffle(order)
    loc, latest = {}, {}
    notes: List[Note] = []
    setup = []
    for j, e in enumerate(ents):
        loc[e] = locs[order[j]]
        key = f"loc.{e}.0"
        notes.append(Note(key=key, value=loc[e], deps=[]))
        latest[e] = key
        setup.append(f"{e} starts in the {loc[e]} room.")
    n_swaps = max(1, (horizon - n_ent) // 2)
    ops = []
    for t in range(1, n_swaps + 1):
        a, b = rng.sample(ents, 2)
        loc[a], loc[b] = loc[b], loc[a]
        ops.append(f"Step {t}: {a} and {b} swap rooms.")
        ka, kb = f"loc.{a}.{t}", f"loc.{b}.{t}"
        pre = [latest[a], latest[b]]
        notes.append(Note(key=ka, value=loc[a], deps=pre))
        notes.append(Note(key=kb, value=loc[b], deps=pre))
        latest[a], latest[b] = ka, kb
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    query = rng.choice(ents)
    ans = loc[query]
    events.append(Answer(value=ans))
    prompt = (
        "Track which room each person is in.\n" + "\n".join(setup) + "\n"
        + "\n".join(ops) + "\n\n"
        "Keep an explicit ledger noting each person's room after every swap as "
        "`loc.<person>.<step> = <room>`.\n"
        f"Which room is {query} in at the end?\n"
        "End with a line: ANSWER: <room color>."
    )
    return LedgerTask(
        family="object_tracking", task_id=f"track_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=ans, gold_facts={n.key: n.value for n in notes}, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 5: Write `scheduling.py` [HELD-OUT]**

```python
# game_theory_llm/reasoning/ledger_tasks/scheduling.py
"""HELD-OUT family (never enters training). Constraint scheduling: each task has a
duration and prerequisites; earliest finish = duration + max(prereq finishes). Ledger
content: `finish.<task>` = time; deps = prereq finish keys. Answer = project completion
time (max finish)."""
from __future__ import annotations

import random
from typing import List

from game_theory_llm.reasoning.ledger_protocol import Answer, Event, LedgerTask, Note, Restate

RESTATE_EVERY = 10


def gen(seed: int, horizon: int) -> LedgerTask:
    rng = random.Random(seed)
    V = max(3, horizon)
    dur, prereq, finish = {}, {}, {}
    notes: List[Note] = []
    lines = []
    for t in range(V):
        d = rng.randint(1, 9)
        dur[t] = d
        k = rng.randint(0, min(2, t))
        prereq[t] = sorted(rng.sample(range(t), k)) if (t > 0 and k > 0) else []
        finish[t] = max((finish[p] for p in prereq[t]), default=0) + d
        notes.append(Note(key=f"finish.{t}", value=str(finish[t]),
                          deps=[f"finish.{p}" for p in prereq[t]]))
        pr = ("after " + ", ".join(f"task {p}" for p in prereq[t])) if prereq[t] else "no prerequisites"
        lines.append(f"Task {t}: duration {d}, {pr}.")
    events: List[Event] = []
    for i, n in enumerate(notes, 1):
        events.append(n)
        if i % RESTATE_EVERY == 0 and i < len(notes):
            events.append(Restate())
    ans = str(max(finish.values()))
    events.append(Answer(value=ans))
    prompt = (
        "Project scheduling. Each task takes its duration and can start only after all its "
        "prerequisites finish (start at time 0 if none).\n" + "\n".join(lines) + "\n\n"
        "Compute each task's earliest finish time in order. Keep an explicit ledger noting "
        "`finish.<task> = <time>`.\n"
        "Report the earliest time ALL tasks are finished (the project completion time).\n"
        "End with a line: ANSWER: <project completion time>."
    )
    return LedgerTask(
        family="scheduling", task_id=f"sched_h{horizon}_s{seed}", prompt=prompt,
        gold_answer=ans, gold_facts={n.key: n.value for n in notes}, events=events,
        horizon=len(notes), needs_ledger=True,
    )
```

- [ ] **Step 6: Run to verify it passes**

Run: `python3 -m pytest tests/test_ledger_tasks.py -v`
Expected: PASS — 24 passed (6 families × [3 horizons + determinism + roundtrip]).

- [ ] **Step 7: Commit**

```bash
git add game_theory_llm/reasoning/ledger_tasks/forward_chain.py \
        game_theory_llm/reasoning/ledger_tasks/object_tracking.py \
        game_theory_llm/reasoning/ledger_tasks/scheduling.py tests/test_ledger_tasks.py
git commit -m "feat(tier5): forward-chain + held-out object-tracking & scheduling generators"
```

---

## Task 5: `naturalize_traces.py` — Sonnet rephrase + extract-back round-trip QC

**Files:**
- Create: `scripts/naturalize_traces.py`
- Test: `tests/test_naturalize_qc.py` (pure QC logic, no network)

**Interfaces:**
- Consumes: `ledger_protocol.render_canonical`, `ledger_tasks.get_generator` (for the smoke `--smoke` path). OpenRouter via `openai`; `OPENROUTER_API_KEY` from the sourced main-repo .env.
- Produces: `qc_ok(ex_notes: set, ex_answer: str, gold_facts: dict, gold_answer: str) -> bool`; `naturalize_all(items: List[dict], workers: int = 8) -> Tuple[List[dict], float]` where each `item` has keys `prompt, canonical, gold_facts, gold_answer, family, style_idx` and each result has `prompt, completion (str|None), ok (bool), family, style`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test (QC logic only — network isolated)**

```python
# tests/test_naturalize_qc.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from naturalize_traces import qc_ok


def test_qc_accepts_exact_set_and_answer_any_order():
    gold_facts = {"a": "1", "b": "2"}
    assert qc_ok({("b", "2"), ("a", "1")}, "3", gold_facts, "3") is True


def test_qc_rejects_altered_value():
    assert qc_ok({("a", "1"), ("b", "9")}, "3", {"a": "1", "b": "2"}, "3") is False


def test_qc_rejects_wrong_answer():
    assert qc_ok({("a", "1")}, "4", {"a": "1"}, "3") is False


def test_qc_rejects_missing_or_extra_note():
    assert qc_ok({("a", "1")}, "3", {"a": "1", "b": "2"}, "3") is False
    assert qc_ok({("a", "1"), ("b", "2"), ("c", "9")}, "3", {"a": "1", "b": "2"}, "3") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_naturalize_qc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'naturalize_traces'`.

- [ ] **Step 3: Write `scripts/naturalize_traces.py`**

```python
# scripts/naturalize_traces.py
"""Sonnet naturalization of canonical ledger traces + extract-back round-trip QC.

RUN WITH SYSTEM python3, after loading the MAIN repo .env (never print the key):
  set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
  python3 scripts/naturalize_traces.py --smoke

Rephraser + extractor model: anthropic/claude-sonnet-4-6 (OpenRouter). Every naturalized
trace must round-trip: extract-back the (key,value) facts + answer and mechanically check
SET equality vs the gold skeleton (order NOT enforced). Reject-and-regenerate (max 2
retries, rotating the style seed) then drop. Corrupted traces would teach corrupted
bookkeeping, so QC is non-negotiable."""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

MODEL = "anthropic/claude-sonnet-4-6"
STYLE_SEEDS = [
    "an engineer's log",
    "a formal derivation",
    "casual working notes",
    "a markdown state table",
    "numbered facts F1/F2/F3...",
]

REPHRASE_SYSTEM = (
    "You rewrite structured reasoning logs into natural, varied working notes. You MUST "
    "preserve every fact exactly: never change any key, value, or dependency; never add or "
    "drop a fact; keep the final answer identical. Only the surface wording and structure "
    "may change. Do NOT solve anything yourself — only restyle what is given."
)

REPHRASE_USER = (
    "Rewrite the following solution log in the style of: {style}.\n"
    "Preserve every noted fact (key = value) and the final answer EXACTLY. You may reorder "
    "prose and reword freely, but every key and its value must still be clearly stated, and "
    "each fact's dependencies (what it was computed from) must remain recoverable. Keep a "
    "final line that states the answer.\n\n"
    "---\n{canonical}\n---\n\n"
    "Return ONLY the rewritten log."
)

EXTRACT_SYSTEM = (
    "You extract structured facts from a reasoning log. Respond ONLY with a JSON object, no "
    "prose, exactly: {\"notes\": [[key, value], ...], \"answer\": \"...\"}. List every fact "
    "the log establishes as [key, value]. If a fact was corrected during the log, report the "
    "FINAL corrected value. 'answer' is the final answer stated in the log."
)

EXTRACT_USER = "Extract the facts and final answer from this log as strict JSON.\n\n---\n{text}\n---"


def qc_ok(ex_notes: set, ex_answer: str, gold_facts: dict, gold_answer: str) -> bool:
    gold = {(str(k).strip(), str(v).strip()) for k, v in gold_facts.items()}
    return ex_notes == gold and str(ex_answer).strip() == str(gold_answer).strip()


def _client():
    import openai
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set — source the main-repo .env first")
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _rephrase(client, canonical: str, style: str) -> str:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7, max_tokens=2048,
        messages=[{"role": "system", "content": REPHRASE_SYSTEM},
                  {"role": "user", "content": REPHRASE_USER.format(style=style, canonical=canonical)}])
    return r.choices[0].message.content or ""


def _extract_back(client, text: str) -> Tuple[set, str]:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=2048,
        messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                  {"role": "user", "content": EXTRACT_USER.format(text=text)}])
    raw = (r.choices[0].message.content or "{}").strip()
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    raw = raw.rstrip("`").strip()
    try:
        d = json.loads(raw)
        notes = {(str(k).strip(), str(v).strip()) for k, v in d.get("notes", [])}
        return notes, str(d.get("answer", "")).strip()
    except Exception:
        return set(), ""


def _naturalize_one(client, item: dict) -> dict:
    for attempt in range(3):        # 1 try + 2 retries
        style = STYLE_SEEDS[(item["style_idx"] + attempt) % len(STYLE_SEEDS)]
        try:
            text = _rephrase(client, item["canonical"], style)
            exn, exa = _extract_back(client, text)
        except Exception:
            continue
        if qc_ok(exn, exa, item["gold_facts"], item["gold_answer"]):
            return {"prompt": item["prompt"], "completion": text, "ok": True,
                    "family": item["family"], "style": style}
    return {"prompt": item["prompt"], "completion": None, "ok": False, "family": item["family"]}


def naturalize_all(items: List[dict], workers: int = 8) -> Tuple[List[dict], float]:
    client = _client()
    out: List[Optional[dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_naturalize_one, client, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    results = [r for r in out if r is not None]
    passed = sum(1 for r in results if r["ok"])
    return results, (passed / len(items) if items else 0.0)


def _smoke():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from game_theory_llm.reasoning.ledger_protocol import render_canonical
    from game_theory_llm.reasoning.ledger_tasks import get_generator
    rng = random.Random(0)
    items = []
    for i, fam in enumerate(["trees", "register_machine", "graph_search", "forward_chain",
                             "trees", "register_machine", "graph_search", "forward_chain",
                             "trees", "graph_search"]):
        t = get_generator(fam)(seed=1000 + i, horizon=rng.choice([12, 18, 24]))
        items.append({"prompt": t.prompt, "canonical": render_canonical(t),
                      "gold_facts": t.gold_facts, "gold_answer": t.gold_answer,
                      "family": t.family, "style_idx": i})
    _, rate = naturalize_all(items, workers=8)
    print(f"[naturalize] smoke n={len(items)} QC pass-rate={rate:.3f}")
    assert rate >= 0.70, f"QC pass-rate {rate:.3f} < 0.70 — investigate rephraser fidelity"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
```

- [ ] **Step 4: Run the QC unit test to verify it passes**

Run: `python3 -m pytest tests/test_naturalize_qc.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Live smoke on 10 traces (paid, ~$0.20)**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
python3 scripts/naturalize_traces.py --smoke
```
Expected: a line `[naturalize] smoke n=10 QC pass-rate=0.XXX` with pass-rate ≥ 0.70 (assertion passes). If below 0.70, tighten `REPHRASE_SYSTEM`/`EXTRACT_SYSTEM` before proceeding — do NOT lower the threshold.

- [ ] **Step 6: Commit**

```bash
git add scripts/naturalize_traces.py tests/test_naturalize_qc.py
git commit -m "feat(tier5): Sonnet naturalization pipeline + extract-back round-trip QC"
```

---

## Task 6: `build_tier5_ledger.py` — full dataset + eval sets + RL pool

**Files:**
- Create: `scripts/build_tier5_ledger.py`
- Test: `tests/test_build_tier5.py` (dry-run composition, no network)

**Interfaces:**
- Consumes: `ledger_tasks.get_generator`, `ledger_protocol.render_canonical`/`inject_recovery`, `naturalize_traces.naturalize_all`.
- Produces (all under `data/runs/tier5/`): `train_tier5.jsonl` ((prompt, completion) pairs); `eval_indomain_<family>_h<H>.jsonl` incl. extrapolated horizons; `eval_heldout_object_tracking.jsonl`, `eval_heldout_scheduling.jsonl` (n=100 each); `rl_pool.jsonl` (rows `{prompt, gold_answer, gold_facts, family, horizon}`). Eval rows: `{story_id, prompt, answer, family, horizon, max_new_tokens}`. Consumed by Tasks 7–11.
- Function `build_traces(dry_run: bool) -> dict` returns composition stats (for the test).

- [ ] **Step 1: Write the failing test (dry-run: skeletons only, no Sonnet calls)**

```python
# tests/test_build_tier5.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_tier5_ledger import build_traces


def test_dryrun_composition_matches_spec():
    stats = build_traces(dry_run=True)
    assert stats["n_long"] == 2560                # 4 families x 640
    assert abs(stats["frac_recovery"] - 0.12) < 0.03
    assert abs(stats["frac_short_noledger"] - 0.15) < 0.03
    assert stats["n_indomain_eval_sets"] >= 4     # >=1 extrapolated set per train family
    assert stats["n_heldout_each"] == 100
    # every training completion ends with an ANSWER line and every eval carries a gold answer
    assert stats["all_train_have_answer_tail"] is True
    assert stats["all_eval_have_gold"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_build_tier5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_tier5_ledger'`.

- [ ] **Step 3: Write `scripts/build_tier5_ledger.py`**

```python
# scripts/build_tier5_ledger.py
"""Build the full Tier-5 ledger dataset + eval sets + RL pool (spec §4 composition).

Composition:
  - 2560 long-horizon train traces (4 train families x 640; horizons 10-90; trees d3-d6),
    ~12% get inject_recovery.
  - render all canonical; naturalize a 75% slice via naturalize_traces.naturalize_all
    (paid ~$60-100; QC pass-rate reported), 25% stay canonical for the Phase-2 dense reward.
  - 450 short no-ledger direct-answer traces (~15%; teach WHEN to deploy).
Eval sets: in-domain per family at trained + EXTRAPOLATED horizons (trees d7=127; register
N=150; graph/chain scaled beyond training); held-out object_tracking + scheduling (n=100).

RUN WITH SYSTEM python3 after sourcing the main-repo .env (naturalization is paid):
  set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
  python3 scripts/build_tier5_ledger.py --build
Dry-run (skeletons only, no Sonnet): python3 scripts/build_tier5_ledger.py --dry-run"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from game_theory_llm.reasoning.ledger_protocol import inject_recovery, render_canonical
from game_theory_llm.reasoning.ledger_tasks import get_generator

OUT = Path("data/runs/tier5")
TRAIN_FAMILIES = ["trees", "register_machine", "graph_search", "forward_chain"]
PER_FAMILY = 640
RECOVERY_FRAC = 0.12
CANON_FRAC = 0.25
N_SHORT = 450
# trained horizons per family (trees map to depths d3-d6)
TRAIN_HORIZONS = {
    "trees": [7, 15, 31, 63],
    "register_machine": [12, 24, 40, 60, 80, 90],
    "graph_search": [12, 24, 40, 60, 80, 90],
    "forward_chain": [12, 24, 40, 60, 80, 90],
}
# extrapolated eval horizons (beyond training) + a couple trained anchors
EVAL_HORIZONS = {
    "trees": [31, 63, 127],            # 127 = d7, extrapolated
    "register_machine": [60, 90, 150],
    "graph_search": [60, 90, 130],
    "forward_chain": [60, 90, 130],
}
EVAL_N = 60
HELDOUT_N = 100
_MAXTOK = lambda h: min(8192, 2048 + 64 * int(h))


def _short_completion(task) -> str:
    return f"This is short enough to answer directly.\nANSWER: {task.gold_answer}"


def _make_long_trace(family, seed, horizon, do_recovery, rng):
    task = get_generator(family)(seed=seed, horizon=horizon)
    events = inject_recovery(task.events, rng) if do_recovery else task.events
    task.events = events
    canonical = render_canonical(task)
    return task, canonical


def build_traces(dry_run: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    long_specs = []            # (task, canonical, is_canonical_slice)
    n_recovery = 0
    for family in TRAIN_FAMILIES:
        hs = TRAIN_HORIZONS[family]
        for i in range(PER_FAMILY):
            seed = i
            horizon = hs[i % len(hs)]
            do_rec = (rng.random() < RECOVERY_FRAC)
            n_recovery += int(do_rec)
            task, canonical = _make_long_trace(family, seed, horizon, do_rec, rng)
            is_canon = (rng.random() < CANON_FRAC)
            long_specs.append((task, canonical, is_canon))

    # short no-ledger traces (mixed families, small horizon)
    short_rows = []
    for j in range(N_SHORT):
        family = TRAIN_FAMILIES[j % len(TRAIN_FAMILIES)]
        task = get_generator(family)(seed=90000 + j, horizon=rng.choice([2, 3]))
        task.needs_ledger = False
        short_rows.append({"prompt": task.prompt, "completion": _short_completion(task)})

    n_long = len(long_specs)
    stats = {
        "n_long": n_long,
        "frac_recovery": n_recovery / n_long,
        "n_canonical": sum(1 for _, _, c in long_specs if c),
        "n_naturalize": sum(1 for _, _, c in long_specs if not c),
        "n_short_noledger": len(short_rows),
        "frac_short_noledger": len(short_rows) / (n_long + len(short_rows)),
        "all_train_have_answer_tail": all(
            r["completion"].rstrip().splitlines()[-1].startswith("ANSWER:")
            for r in short_rows) and all(
            c.rstrip().splitlines()[-1].startswith("ANSWER:") for _, c, _ in long_specs),
    }

    # build eval + rl pool (skeleton-only; always runs, cheap)
    n_eval_sets = 0
    rl_pool = []
    all_eval_have_gold = True
    for family in TRAIN_FAMILIES:
        for H in EVAL_HORIZONS[family]:
            rows = []
            for k in range(EVAL_N):
                t = get_generator(family)(seed=100000 + k, horizon=H)
                rows.append({"story_id": t.task_id, "prompt": t.prompt, "answer": t.gold_answer,
                             "family": family, "horizon": t.horizon, "max_new_tokens": _MAXTOK(H)})
                rl_pool.append({"prompt": t.prompt, "gold_answer": t.gold_answer,
                                "gold_facts": t.gold_facts, "family": family, "horizon": t.horizon})
                all_eval_have_gold &= bool(t.gold_answer)
            n_eval_sets += 1
            if not dry_run:
                _write(OUT / f"eval_indomain_{family}_h{H}.jsonl", rows)
    for family in ["object_tracking", "scheduling"]:
        rows = []
        for k in range(HELDOUT_N):
            t = get_generator(family)(seed=200000 + k, horizon=rng.choice([30, 50, 70]))
            rows.append({"story_id": t.task_id, "prompt": t.prompt, "answer": t.gold_answer,
                         "family": family, "horizon": t.horizon, "max_new_tokens": _MAXTOK(70)})
            all_eval_have_gold &= bool(t.gold_answer)
        if not dry_run:
            _write(OUT / f"eval_heldout_{family}.jsonl", rows)
    stats["n_indomain_eval_sets"] = n_eval_sets
    stats["n_heldout_each"] = HELDOUT_N
    stats["all_eval_have_gold"] = all_eval_have_gold

    if dry_run:
        return stats

    # naturalize the 75% slice (PAID) and assemble train jsonl
    from naturalize_traces import naturalize_all
    train_rows = []
    for task, canonical, is_canon in long_specs:
        if is_canon:
            train_rows.append({"prompt": task.prompt, "completion": canonical})
    nat_items = [{"prompt": t.prompt, "canonical": c, "gold_facts": t.gold_facts,
                  "gold_answer": t.gold_answer, "family": t.family, "style_idx": i}
                 for i, (t, c, is_canon) in enumerate(long_specs) if not is_canon]
    nat_results, pass_rate = naturalize_all(nat_items, workers=8)
    for r in nat_results:
        if r["ok"]:
            train_rows.append({"prompt": r["prompt"], "completion": r["completion"]})
    train_rows.extend(short_rows)
    random.Random(7).shuffle(train_rows)
    _write(OUT / "train_tier5.jsonl", train_rows)
    _write(OUT / "rl_pool.jsonl", rl_pool)

    stats["naturalize_pass_rate"] = pass_rate
    stats["n_train_rows"] = len(train_rows)
    stats["train_family_mix"] = "canonical+naturalized+short (see build)"
    print(json.dumps({k: v for k, v in stats.items() if k != "all_train_have_answer_tail"}, indent=2))
    print(f"[build] naturalize QC pass-rate = {pass_rate:.3f}")
    print(f"[build] wrote {len(train_rows)} train rows, {len(rl_pool)} rl-pool rows, "
          f"{n_eval_sets} in-domain eval sets, 2 held-out eval sets to {OUT}/")
    return stats


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="full paid build (naturalization)")
    ap.add_argument("--dry-run", action="store_true", help="skeleton composition only, no Sonnet")
    args = ap.parse_args()
    s = build_traces(dry_run=not args.build)
    if args.dry_run or not args.build:
        print(json.dumps(s, indent=2))
```

- [ ] **Step 4: Run the dry-run test to verify it passes**

Run: `python3 -m pytest tests/test_build_tier5.py -v`
Expected: PASS — 1 passed.

- [ ] **Step 5: Dry-run the builder (writes eval + rl_pool, no paid calls)**

Run: `python3 scripts/build_tier5_ledger.py --dry-run`
Expected: JSON stats with `"n_long": 2560`, `frac_recovery≈0.12`, `frac_short_noledger≈0.15`, `"n_indomain_eval_sets": 12`, `"n_heldout_each": 100`, `all_*` true. (Note: `--dry-run` still writes the cheap eval/rl-pool files.)

- [ ] **Step 6: Full paid build (~$60–100; naturalizes the 75% slice)**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
python3 scripts/build_tier5_ledger.py --build
```
Expected: prints composition stats + `[build] naturalize QC pass-rate = 0.XX` (report it; expect ≥ 0.70) + `[build] wrote NNNN train rows ...`. Files land in `data/runs/tier5/`.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_tier5_ledger.py tests/test_build_tier5.py
git commit -m "feat(tier5): dataset builder (train + in-domain/held-out evals + RL pool)"
```
(Data files under `data/runs/tier5/` — commit per repo convention for run artifacts, or leave untracked if large; the plan does not assume they are versioned.)

---

## Task 7: Phase-1 SFT via `scripts/tinker_sft.py` (reuse) on `train_tier5.jsonl`

Per the brief, Phase-1 reuses the EXISTING `scripts/tinker_sft.py` (LoRA SFT on (prompt, completion) JSONL, loss masked to completion, sampler checkpoint saved). Its default `--base-model` is `meta-llama/Llama-3.1-8B-Instruct` and default `--save-name gametree_sft_v1`; pass Tier-5 overrides. One adaptation is needed: the script hardcodes writing the checkpoint path to `data/runs/gametree/sft_checkpoint.txt` — add a `--ckpt-out` arg so Tier-5 writes its own sidecar. (This satisfies spec §8's `tinker_sft_ledger.py` slot without a new launcher.)

**Files:**
- Modify: `scripts/tinker_sft.py` (add `--ckpt-out` arg; use it for the sidecar write)

**Interfaces:**
- Consumes: `data/runs/tier5/train_tier5.jsonl` (Task 6).
- Produces: a Tinker sampler checkpoint named `tier5_sft`; its `tinker://...` path written to `data/runs/tier5/sft_checkpoint.txt` (consumed by Tasks 8, 9, 10).

- [ ] **Step 1: Add `--ckpt-out` to `scripts/tinker_sft.py`**

Add the argparse line (in `main()`, next to `--save-name`):
```python
    ap.add_argument("--ckpt-out", default="data/runs/gametree/sft_checkpoint.txt",
                    help="sidecar file to receive the tinker:// checkpoint path")
```
Replace the final hardcoded write:
```python
    Path("data/runs/gametree/sft_checkpoint.txt").write_text(path)
```
with:
```python
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_out).write_text(path)
```

- [ ] **Step 2: Smoke run on 50 rows (uses existing `--limit`)**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
.venv-tinker/bin/python scripts/tinker_sft.py \
  --train data/runs/tier5/train_tier5.jsonl \
  --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --rank 32 --lr 1e-4 --epochs 1 --batch 8 --max-len 4096 \
  --limit 50 --save-name tier5_sft_smoke \
  --ckpt-out data/runs/tier5/sft_checkpoint_smoke.txt
```
Expected: `[sft] 50 train examples; base=Qwen/Qwen3-30B-A3B-Instruct-2507 rank=32` … `[sft] epoch 1/1 done` … `[sft] saved sampler checkpoint: tinker://...`. Confirms the loss_fn_inputs keys + base-model name resolve on Qwen3-30B before the full run.

- [ ] **Step 3: Full SFT run (checkpoint = `tier5_sft`)**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
.venv-tinker/bin/python scripts/tinker_sft.py \
  --train data/runs/tier5/train_tier5.jsonl \
  --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --rank 32 --lr 1e-4 --epochs 2 --batch 8 --max-len 4096 \
  --save-name tier5_sft \
  --ckpt-out data/runs/tier5/sft_checkpoint.txt
```
Expected: `[sft] ~3000 train examples ...` … `[sft] epoch 1/2 done` … `[sft] epoch 2/2 done` … `[sft] saved sampler checkpoint: tinker://.../tier5_sft/...`, and `data/runs/tier5/sft_checkpoint.txt` now contains that path.

- [ ] **Step 4: Verify the sidecar path is readable**

Run: `cat data/runs/tier5/sft_checkpoint.txt`
Expected: a single `tinker://...` line (the SFT checkpoint).

- [ ] **Step 5: Commit**

```bash
git add scripts/tinker_sft.py
git commit -m "feat(tier5): add --ckpt-out to tinker_sft; Phase-1 SFT on train_tier5"
```

---

## Task 8: Gate eval — base vs `tier5_sft` in-domain; `gate_report.json` + gate decision

`scripts/tinker_eval.py` currently parses `<answer>`/`<decision>` tags only. Add a `ledger` eval kind that mechanically parses a required `ANSWER: <value>` tail (accepting `<answer>` tags too) and normalized-exact-matches `row["answer"]`. Then a gate driver runs base vs SFT on the in-domain eval sets at trained + extrapolated horizons and applies the spec §5 gate.

**Files:**
- Modify: `scripts/tinker_eval.py` (add `ledger` to `--eval` choices; add tail parser + `ledger` scoring branch)
- Create: `scripts/tier5_gate.py`
- Test: `tests/test_ledger_eval_scoring.py`

**Interfaces:**
- Consumes: `data/runs/tier5/eval_indomain_*.jsonl` (Task 6), `data/runs/tier5/sft_checkpoint.txt` (Task 7).
- Produces: `scripts/tinker_eval.py` gains eval kind `ledger`; `data/runs/tier5/gate_report.json` (`{per_setname: {base_acc, sft_acc}, decision}`); prints `GATE DECISION: PROCEED|SKIP-RL|STOP`.

- [ ] **Step 1: Write the failing scoring test**

```python
# tests/test_ledger_eval_scoring.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tinker_eval import score


def _row(ans):
    return {"answer": ans}


def test_ledger_answer_tail_correct():
    text = "LEDGER\nNOTE: a = 1\nANSWER: 42"
    assert score("ledger", text, _row("42")) == (True, True)


def test_ledger_accepts_answer_tag():
    assert score("ledger", "reasoning...\n<answer>7</answer>", _row("7")) == (True, True)


def test_ledger_normalizes_parens_and_case():
    assert score("ledger", "ANSWER: (A)", _row("a"))[0] is True


def test_ledger_wrong_and_unparsed():
    assert score("ledger", "ANSWER: 5", _row("42")) == (False, True)
    assert score("ledger", "no final answer here", _row("42")) == (False, False)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_eval_scoring.py -v`
Expected: FAIL — `ValueError`/assertion because `score` does not handle `"ledger"` (falls through to the legacy A-D branch).

- [ ] **Step 3: Modify `scripts/tinker_eval.py`**

Add `"ledger"` to the `--eval` choices list:
```python
    ap.add_argument("--eval", required=True,
                    choices=["gametree", "gsm8k", "mmlu", "bbh",
                             "freetext", "dyck", "prontoqa", "mmlu_pro", "bbh_hard", "ledger"])
```
Add these module-level helpers next to the other regexes (after `_DEC_J`):
```python
_ANS_TAIL = _re.compile(r"ANSWER:\s*(.+?)\s*$", _re.IGNORECASE | _re.MULTILINE)
_ANS_TAG = _re.compile(r"<answer>\s*(.+?)\s*</answer>", _re.IGNORECASE | _re.DOTALL)


def _ledger_norm(s):
    return s.lower().strip().strip(".()").strip()


def _final_answer(text):
    """Required ANSWER: tail, or an <answer>...</answer> tag; returns the last one or None."""
    tag = _ANS_TAG.findall(text or "")
    if tag:
        return tag[-1].strip()
    tail = _ANS_TAIL.findall(text or "")
    return tail[-1].strip() if tail else None
```
Add a `ledger` branch at the TOP of `score()` (before the `gametree` branch):
```python
    if eval_kind == "ledger":
        pred = _final_answer(text)
        gold = str(row["answer"]).strip()
        return (pred is not None and _ledger_norm(pred) == _ledger_norm(gold), pred is not None)
```

- [ ] **Step 4: Run the scoring test to verify it passes**

Run: `python3 -m pytest tests/test_ledger_eval_scoring.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Write `scripts/tier5_gate.py`**

```python
# scripts/tier5_gate.py
"""Tier-5 gate (spec §5): run base vs tier5_sft on the in-domain eval sets (trained +
EXTRAPOLATED horizons), aggregate accuracy, write gate_report.json, print the decision.

RUN WITH .venv-tinker after sourcing the main-repo .env:
  set -a; source .../.env; set +a
  .venv-tinker/bin/python scripts/tier5_gate.py"""
from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

OUT = Path("data/runs/tier5")
TOK = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PY = ".venv-tinker/bin/python"
EXTRAP = {"trees_h127", "register_machine_h150", "graph_search_h130", "forward_chain_h130"}


def _run(corpus, spec, tag):
    out = OUT / f"gate_{tag}_{Path(corpus).stem}.json"
    subprocess.run([PY, "scripts/tinker_eval.py", "--eval", "ledger", "--corpus", corpus,
                    *spec, "--tokenizer", TOK, "--temperature", "0", "--max-tokens", "4096",
                    "--out", str(out)], check=True)
    return json.loads(out.read_text())["accuracy"]


def main():
    sft_path = (OUT / "sft_checkpoint.txt").read_text().strip()
    sets = sorted(glob.glob(str(OUT / "eval_indomain_*.jsonl")))
    report = {"per_set": {}, "sft_path": sft_path}
    for corpus in sets:
        name = Path(corpus).stem.replace("eval_indomain_", "")
        base = _run(corpus, ["--base-model", TOK], "base")
        sft = _run(corpus, ["--model-path", sft_path], "sft")
        report["per_set"][name] = {"base_acc": base, "sft_acc": sft, "delta": sft - base}
    extrap = [v for n, v in report["per_set"].items() if n in EXTRAP]
    trained = [v for n, v in report["per_set"].items() if n not in EXTRAP]
    mean_extrap_delta = sum(v["delta"] for v in extrap) / max(1, len(extrap))
    mean_sft = sum(v["sft_acc"] for v in report["per_set"].values()) / max(1, len(report["per_set"]))
    mean_trained_sft = sum(v["sft_acc"] for v in trained) / max(1, len(trained))
    if mean_sft >= 0.90:
        decision = "SKIP-RL"       # evals saturated -> go to final eval
    elif mean_extrap_delta >= 0.15 or (mean_trained_sft >= 0.60 and mean_extrap_delta >= 0.05):
        decision = "PROCEED"       # behavior installed, reliability lags -> Phase 2
    else:
        decision = "STOP"          # ledger not installed even in-domain -> report the negative
    report["mean_extrap_delta"] = mean_extrap_delta
    report["mean_sft_acc"] = mean_sft
    report["mean_trained_sft_acc"] = mean_trained_sft
    report["decision"] = decision
    (OUT / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"GATE DECISION: {decision}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the gate**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
.venv-tinker/bin/python scripts/tier5_gate.py
```
Expected: per-set base/sft accuracies printed, `data/runs/tier5/gate_report.json` written, and a final line `GATE DECISION: PROCEED` (or `SKIP-RL` / `STOP`). Record the decision — it drives Task 9.

- [ ] **Step 7: Commit**

```bash
git add scripts/tinker_eval.py scripts/tier5_gate.py tests/test_ledger_eval_scoring.py
git commit -m "feat(tier5): ledger eval kind (ANSWER tail) + Phase-1 gate driver"
```

---

## Task 9: Phase-2 GRPO — `gt_ledger_env.py` + `tinker_grpo_ledger.py`

Mirror `scripts/gt_rl_env.py` + `scripts/tinker_grpo.py`. The env's `check_answer` returns a graded **float** reward `answer_correct + 0.5·ledger_reward` (ProblemEnv.step casts it through `float()`, exactly like `ProcessGameTheoryEnv`), giving dense signal even when final answers fail. The reward math reuses `ledger_protocol.parse_canonical`/`score_claims`; to avoid triggering the heavy `game_theory_llm` package `__init__` in the Tinker venv, the env imports `ledger_protocol` as a standalone top-level module via `sys.path` (the module is pure stdlib). **Gated on Task 8: run only if the gate decision was PROCEED.**

**Files:**
- Create: `scripts/gt_ledger_env.py`
- Create: `scripts/tinker_grpo_ledger.py`
- Test: `tests/test_ledger_reward.py`

**Interfaces:**
- Consumes: `game_theory_llm/reasoning/ledger_protocol.py` (`parse_canonical`, `score_claims`); `data/runs/tier5/rl_pool.jsonl` (Task 6, rows `{prompt, gold_answer, gold_facts, family, horizon}`); `data/runs/tier5/gate_report.json` (cram-boundary horizons); tinker_cookbook `ProblemEnv`/`RLDataset` surfaces (as in `gt_rl_env.py`).
- Produces: `LedgerEnv`, `LedgerDataset`, `LedgerDatasetBuilder(train_path, batch_size, group_size, model_name_for_tokenizer, renderer_name, families="", horizons="")`; a GRPO run + checkpoint.

- [ ] **Step 1: Write the failing reward test (pure protocol math — no tinker import)**

```python
# tests/test_ledger_reward.py
from game_theory_llm.reasoning.ledger_protocol import parse_canonical, score_claims


def _reward(text, gold_facts, gold_answer):
    notes, ans = parse_canonical(text)
    s = score_claims(notes, ans, gold_facts, gold_answer)
    return float(s["answer_correct"]) + 0.5 * s["ledger_reward"]


def test_perfect_completion_scores_1_5():
    text = "LEDGER\nNOTE: a = 1\nNOTE: b = 2\nANSWER: 3"
    assert _reward(text, {"a": "1", "b": "2"}, "3") == 1.5


def test_wrong_everything_scores_0():
    text = "LEDGER\nNOTE: a = 9\nANSWER: 7"
    assert _reward(text, {"a": "1", "b": "2"}, "3") == 0.0


def test_right_answer_partial_ledger_between_0_and_1_5():
    text = "LEDGER\nNOTE: a = 1\nANSWER: 3"     # answer right, half the notes
    r = _reward(text, {"a": "1", "b": "2"}, "3")
    assert 1.0 < r < 1.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ledger_reward.py -v`
Expected: FAIL only if protocol regressed; since Task 1 landed `parse_canonical`/`score_claims`, this test should already PASS — treat it as the reward-contract lock for the env below. If it does not pass, fix the env reward math to match, not the test.

- [ ] **Step 3: Write `scripts/gt_ledger_env.py`**

```python
# scripts/gt_ledger_env.py
"""GRPO env + dataset for Tier-5 inline-ledger RL (mirrors gt_rl_env.py).

Imported ONLY in the Tinker 3.11 venv (depends on tinker_cookbook). Reward =
answer_correct + 0.5*ledger_reward, computed via ledger_protocol.parse_canonical +
score_claims. ledger_protocol is imported as a STANDALONE top-level module (it is pure
stdlib) so we never trigger the heavy game_theory_llm package __init__ in this venv."""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections.abc import Sequence
from functools import partial

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "game_theory_llm" / "reasoning"))
import ledger_protocol as lp  # noqa: E402  standalone import; does NOT load game_theory_llm/__init__

CANON_ANCHOR = (
    "Keep an explicit running ledger. For each fact write a line `NOTE: <key> = <value>`; "
    "restate known facts on `CHECKPOINT:` lines periodically; finish with `ANSWER: <value>`."
)


class LedgerEnv(ProblemEnv):
    def __init__(self, problem: dict, renderer, convo_prefix=None, format_coef: float = 0.1):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.problem = problem

    def get_question(self) -> str:
        return self.problem["prompt"] + "\n\n" + CANON_ANCHOR

    def check_format(self, sample_str: str) -> bool:
        _, ans = lp.parse_canonical(sample_str)
        return ans is not None

    def check_answer(self, sample_str: str) -> float:  # type: ignore[override]
        notes, ans = lp.parse_canonical(sample_str)
        sc = lp.score_claims(notes, ans, self.problem["gold_facts"], self.problem["gold_answer"])
        return float(sc["answer_correct"]) + 0.5 * sc["ledger_reward"]

    def get_reference_answer(self) -> str:
        return str(self.problem["gold_answer"])


class LedgerDataset(RLDataset):
    def __init__(self, rows, batch_size, group_size, renderer):
        self.rows, self.batch_size, self.group_size, self.renderer = rows, batch_size, group_size, renderer

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        return [ProblemGroupBuilder(
            env_thunk=partial(LedgerEnv, self.rows[i], self.renderer),
            num_envs=self.group_size) for i in range(start, end)]


@chz.chz
class LedgerDatasetBuilder(RLDatasetBuilder):
    train_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    families: str = ""      # comma-separated filter (cram-boundary selection)
    horizons: str = ""      # comma-separated int filter

    async def __call__(self):
        rows = [json.loads(l) for l in open(self.train_path) if l.strip()]
        fam = {f for f in self.families.split(",") if f}
        hor = {int(h) for h in self.horizons.split(",") if h}
        if fam:
            rows = [r for r in rows if r["family"] in fam]
        if hor:
            rows = [r for r in rows if int(r["horizon"]) in hor]
        renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
        return LedgerDataset(rows, self.batch_size, self.group_size, renderer), None


if __name__ == "__main__":       # self-check runnable under .venv-tinker
    good = "LEDGER\nNOTE: a = 1\nNOTE: b = 2\nANSWER: 3"
    prob = {"prompt": "p", "gold_facts": {"a": "1", "b": "2"}, "gold_answer": "3"}
    notes, ans = lp.parse_canonical(good)
    sc = lp.score_claims(notes, ans, prob["gold_facts"], prob["gold_answer"])
    assert float(sc["answer_correct"]) + 0.5 * sc["ledger_reward"] == 1.5
    print("gt_ledger_env self-check OK")
```

- [ ] **Step 4: Write `scripts/tinker_grpo_ledger.py`**

```python
# scripts/tinker_grpo_ledger.py
"""GRPO on Tier-5 inline-ledger tasks via the Tinker cookbook rl.train loop.

RUN WITH:  .venv-tinker/bin/python scripts/tinker_grpo_ledger.py [--smoke] \
             --model-path <SFT tinker:// ckpt> --families <...> --horizons <...>
(source the main-repo .env first). Cram-boundary --families/--horizons come from
data/runs/tier5/gate_report.json (horizons where the SFT groups are mixed)."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_ledger_env import LedgerDatasetBuilder  # noqa: E402

from tinker_cookbook import checkpoint_utils  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402


async def amain(a):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=a.model, explicit_renderer_name=None, load_checkpoint_path=a.model_path or None)
    kl_ref = train.KLReferenceConfig(base_model=a.model) if a.kl > 0 else None
    config = train.Config(
        learning_rate=a.lr, model_name=a.model, max_tokens=a.max_tokens,
        kl_penalty_coef=a.kl, kl_reference_config=kl_ref, renderer_name=renderer_name,
        load_checkpoint_path=a.model_path or None,
        log_path=a.log_path, eval_every=a.eval_every, save_every=a.save_every,
        dataset_builder=LedgerDatasetBuilder(
            train_path=a.train, batch_size=a.groups_per_batch, group_size=a.group_size,
            model_name_for_tokenizer=a.model, renderer_name=renderer_name,
            families=a.families, horizons=a.horizons))
    await train.main(config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--model-path", default="", help="SFT tinker:// checkpoint to warm-start")
    ap.add_argument("--train", default="data/runs/tier5/rl_pool.jsonl")
    ap.add_argument("--families", default="")
    ap.add_argument("--horizons", default="")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group-size", type=int, default=8, dest="group_size")
    ap.add_argument("--groups-per-batch", type=int, default=64, dest="groups_per_batch")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=10, dest="save_every")
    ap.add_argument("--log-path", default="data/runs/tier5/grpo_ledger_run", dest="log_path")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.groups_per_batch, a.group_size = 4, 4
        a.eval_every, a.save_every, a.max_tokens = 0, 2, 1024
    asyncio.run(amain(a))
```

- [ ] **Step 5: Run the env self-check + reward test**

Run:
```bash
python3 -m pytest tests/test_ledger_reward.py -v
.venv-tinker/bin/python scripts/gt_ledger_env.py
```
Expected: `3 passed` and `gt_ledger_env self-check OK`.

- [ ] **Step 6: Smoke GRPO, then full run (ONLY if gate = PROCEED)**

Read the cram-boundary from the gate report, e.g. `--families register_machine,graph_search --horizons 90,130`, then:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
SFT=$(cat data/runs/tier5/sft_checkpoint.txt)
.venv-tinker/bin/python scripts/tinker_grpo_ledger.py --smoke --model-path "$SFT" \
  --families register_machine,graph_search --horizons 90
# then the full run:
.venv-tinker/bin/python scripts/tinker_grpo_ledger.py --model-path "$SFT" \
  --families register_machine,graph_search,forward_chain --horizons 90,130 \
  --save-every 10 --log-path data/runs/tier5/grpo_ledger_run
```
Expected: cookbook GRPO logs (per-step mean reward rising off the SFT warm-start; dense reward present even when answer accuracy is low) and periodic checkpoints saved under the log path. Record the final RL checkpoint `tinker://...` into `data/runs/tier5/rl_checkpoint.txt`.

- [ ] **Step 7: Commit**

```bash
git add scripts/gt_ledger_env.py scripts/tinker_grpo_ledger.py tests/test_ledger_reward.py
git commit -m "feat(tier5): Phase-2 GRPO ledger env + launcher (dense P*R reward)"
```

---

## Task 10: Generalization battery part 1 — BBH fetch + `tier5_evalsuite.sh`

`data/runs/bbh/` currently holds ONLY `logical_deduction_eval.jsonl`. Fetch the three long-horizon state-tracking BBH subsets from HF `maveriq/bigbenchhard` into the existing bbh jsonl schema (plus an `answer` field required by the `ledger` scorer). Then an eval-suite runs base / SFT / SFT+RL across in-domain extrapolation, held-out families, the 3 BBH sets, and the GSM8K control.

**Files:**
- Create: `scripts/fetch_bbh_tier5.py`
- Create: `scripts/tier5_evalsuite.sh`

**Interfaces:**
- Consumes: HF `maveriq/bigbenchhard` (configs `multistep_arithmetic_two`, `tracking_shuffled_objects_three_objects`, `dyck_languages`); `data/runs/tier5/eval_indomain_*_h{127,150,130}.jsonl`, `data/runs/tier5/eval_heldout_*.jsonl`; `data/runs/capability/gsm8k_eval.jsonl`; `data/runs/tier5/{sft,rl}_checkpoint.txt`.
- Produces: `data/runs/bbh/{multistep_arithmetic_two,tracking_shuffled_objects_three_objects,dyck_languages}_eval.jsonl` (schema `{story_id, prompt, answer, seed, game_type, framing, max_new_tokens}`); eval outputs `data/runs/tier5/t5_<tag>_<name>.json`.

- [ ] **Step 1: Write `scripts/fetch_bbh_tier5.py`**

```python
# scripts/fetch_bbh_tier5.py
"""Fetch 3 long-horizon BBH subsets into data/runs/bbh/ in the eval schema used by
scripts/tinker_eval.py --eval ledger (mechanical ANSWER: tail; gold in row['answer']).

RUN WITH:  .venv-tinker/bin/python scripts/fetch_bbh_tier5.py
(datasets is available in .venv-tinker 4.8.5). Answers are compared with normalized
exact-match; for dyck we store the closer sequence with whitespace removed and instruct
the model to output it without spaces."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

SUBSETS = ["multistep_arithmetic_two", "tracking_shuffled_objects_three_objects", "dyck_languages"]
TAIL = "\n\nThink step by step, then end with a line: ANSWER: <final answer>."
OUT = Path("data/runs/bbh")


def _load(subset):
    ds = load_dataset("maveriq/bigbenchhard", subset)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        ds = _load(subset)
        rows = []
        for i, ex in enumerate(ds):
            gold = str(ex["target"]).strip()
            prompt_body = ex["input"].strip()
            if subset == "dyck_languages":
                gold = gold.replace(" ", "")
                prompt_body += ("\n\nGive ONLY the sequence of closing brackets needed to "
                                "close all open brackets, with NO spaces.")
            rows.append({
                "story_id": f"bbh_{subset}_{i:04d}",
                "prompt": prompt_body + TAIL,
                "answer": gold,
                "seed": i,
                "game_type": f"bbh_{subset}",
                "framing": subset,
                "max_new_tokens": 2048,
            })
        out = OUT / f"{subset}_eval.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[fetch_bbh] {subset}: wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fetch**

Run: `.venv-tinker/bin/python scripts/fetch_bbh_tier5.py`
Expected: three lines `[fetch_bbh] <subset>: wrote NNN rows -> data/runs/bbh/<subset>_eval.jsonl` (each subset ~250 rows).

- [ ] **Step 3: Write `scripts/tier5_evalsuite.sh`**

```bash
#!/bin/bash
# Tier-5 generalization battery. Usage:
#   bash scripts/tier5_evalsuite.sh base
#   bash scripts/tier5_evalsuite.sh sft  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt)"
#   bash scripts/tier5_evalsuite.sh rl   --model-path "$(cat data/runs/tier5/rl_checkpoint.txt)"
# For base, pass:  base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
set -e
TAG="$1"; shift; SPEC="$@"
TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/tier5; B=data/runs/bbh
run(){ $PY scripts/tinker_eval.py --eval "$1" --corpus "$2" $SPEC --tokenizer $TOK \
        --temperature 0 --max-tokens "${4:-4096}" --out "$D/t5_${TAG}_$3.json" \
        2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"; }
# 1) in-domain extrapolated horizons
run ledger $D/eval_indomain_trees_h127.jsonl            indom_trees_d7 4096
run ledger $D/eval_indomain_register_machine_h150.jsonl indom_reg_n150 6144
run ledger $D/eval_indomain_graph_search_h130.jsonl     indom_graph_130 6144
run ledger $D/eval_indomain_forward_chain_h130.jsonl    indom_chain_130 6144
# 2) held-out families (zero-shot)
run ledger $D/eval_heldout_object_tracking.jsonl        heldout_tracking 4096
run ledger $D/eval_heldout_scheduling.jsonl             heldout_scheduling 4096
# 3) real benchmarks (>=3) + short-form control
run ledger $B/multistep_arithmetic_two_eval.jsonl             bbh_arith 2048
run ledger $B/tracking_shuffled_objects_three_objects_eval.jsonl bbh_track 2048
run ledger $B/dyck_languages_eval.jsonl                       bbh_dyck 2048
run gsm8k  data/runs/capability/gsm8k_eval.jsonl              gsm8k_control 1536
echo "[tier5-suite] $TAG done"
```

- [ ] **Step 4: Run the battery for all three models**

Run:
```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
bash scripts/tier5_evalsuite.sh base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
bash scripts/tier5_evalsuite.sh sft  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt)"
# rl only if Task 9 ran:
bash scripts/tier5_evalsuite.sh rl   --model-path "$(cat data/runs/tier5/rl_checkpoint.txt)"
```
Expected: each `run` prints `[eval] ledger: n=... accuracy=... parse_rate=...` and writes `data/runs/tier5/t5_<tag>_<name>.json`; each suite ends `[tier5-suite] <tag> done`.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_bbh_tier5.py scripts/tier5_evalsuite.sh
git commit -m "feat(tier5): BBH fetch + generalization eval battery"
```

---

## Task 11: Generalization battery part 2 — attribution (invariant judge + suppression ablation)

Two attributions (spec §7.4): (a) **unprompted technique-usage rate** in transfer domains — an LLM-judge classification against the five §3 invariants (Sonnet; never regex); (b) **suppression ablation** — a system prompt forbidding explicit state-tracking; gains should shrink toward base. This needs two small `tinker_eval.py` additions: `--system` (inject a system message) and `--save-raw` (dump raw completions for the judge).

**Files:**
- Modify: `scripts/tinker_eval.py` (add `--system` and `--save-raw`)
- Create: `scripts/tier5_attribution.py`
- Test: `tests/test_invariant_judge_parse.py` (JSON-parse logic, no network)

**Interfaces:**
- Consumes: raw-completion jsonl from `tinker_eval.py --save-raw` (rows `{story_id, family, prompt, text}`); the `t5_<tag>_*.json` eval outputs from Task 10.
- Produces: `judge_one(client, text) -> dict` (six booleans); `scripts/tier5_attribution.py` subcommands `judge` (usage-rate report) and `suppression` (gain-shrinkage report); prints rates and writes `data/runs/tier5/attribution.json`.

- [ ] **Step 1: Add `--system` + `--save-raw` to `scripts/tinker_eval.py`**

Add argparse args (next to `--out`):
```python
    ap.add_argument("--system", default=None, help="optional system message (e.g., suppression)")
    ap.add_argument("--save-raw", default=None, help="jsonl path to dump raw completions")
```
In `run_one`, build messages with the optional system turn and include `text` in the return:
```python
        msgs = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": row["prompt"]}]
        text_in = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text_in, add_special_tokens=False).input_ids
```
(replace the two lines that built `text`/`ids` from a user-only message), and add `"text": out` to the returned dict. After the results are computed in `main`, dump raw if requested:
```python
    if args.save_raw:
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_raw, "w") as f:
            for r in res:
                f.write(json.dumps({"story_id": r["story_id"], "family": r["family"],
                                    "text": r.get("text", "")}) + "\n")
```
(so `run_one` must also carry `"text": out` through — add it to the returned dict.)

- [ ] **Step 2: Write the failing judge-parse test**

```python
# tests/test_invariant_judge_parse.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tier5_attribution import _parse_judge


def test_parse_valid_json():
    raw = ('{"externalizes": true, "grounds_steps": true, "small_steps": false, '
           '"restates": false, "reads_off_answer": true, "uses_ledger": true}')
    d = _parse_judge(raw)
    assert d["uses_ledger"] is True and d["small_steps"] is False


def test_parse_garbage_defaults_false():
    d = _parse_judge("not json")
    assert all(v is False for v in d.values())
    assert set(d) == {"externalizes", "grounds_steps", "small_steps",
                      "restates", "reads_off_answer", "uses_ledger"}
```

- [ ] **Step 3: Run to verify it fails**

Run: `python3 -m pytest tests/test_invariant_judge_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tier5_attribution'`.

- [ ] **Step 4: Write `scripts/tier5_attribution.py`**

```python
# scripts/tier5_attribution.py
"""Tier-5 attribution: (a) LLM-judge invariant-usage rate on transfer-domain outputs;
(b) suppression-ablation gain shrinkage.

RUN WITH SYSTEM python3 after sourcing the main-repo .env (judge = Sonnet via OpenRouter):
  set -a; source .../.env; set +a
  python3 scripts/tier5_attribution.py judge --samples data/runs/tier5/raw_sft_heldout_tracking.jsonl
  python3 scripts/tier5_attribution.py suppression \
      --normal data/runs/tier5/t5_sft_heldout_tracking.json \
      --suppressed data/runs/tier5/t5_sftsupp_heldout_tracking.json
Never print the API key. Invariant classification is an LLM judgment (project rule: no
regex for quality/semantic judgments)."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "anthropic/claude-sonnet-4-6"
INVARIANTS = ["externalizes", "grounds_steps", "small_steps", "restates",
              "reads_off_answer", "uses_ledger"]

JUDGE_SYSTEM = (
    "You classify whether a model's solution uses an explicit inline state-tracking "
    "(\"ledger\") technique, judged by five semantic invariants. Respond ONLY with a JSON "
    "object, no prose, with exactly these boolean keys:\n"
    "  externalizes      — establishes facts are written down compactly as they are found, "
    "not held implicitly.\n"
    "  grounds_steps     — each step states what is computable now and from which recorded facts.\n"
    "  small_steps       — proceeds in small bounded steps rather than re-deriving long chains.\n"
    "  restates          — periodically re-states the full known-state (a checkpoint/summary).\n"
    "  reads_off_answer  — the final answer is read off the recorded state, not recomputed from scratch.\n"
    "  uses_ledger       — true iff the response overall clearly uses the explicit ledger technique.\n"
    'Example: {"externalizes": true, "grounds_steps": true, "small_steps": true, '
    '"restates": false, "reads_off_answer": true, "uses_ledger": true}'
)

JUDGE_USER = "MODEL SOLUTION:\n{text}\n\nClassify against the five invariants. JSON only."


def _parse_judge(raw: str) -> dict:
    out = {k: False for k in INVARIANTS}
    try:
        s = (raw or "").strip()
        for fence in ("```json", "```"):
            if s.startswith(fence):
                s = s[len(fence):]
        s = s.rstrip("`").strip()
        d = json.loads(s)
        for k in INVARIANTS:
            out[k] = bool(d.get(k, False))
    except Exception:
        pass
    return out


def judge_one(client, text: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=128,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": JUDGE_USER.format(text=text[:6000])}])
    return _parse_judge(r.choices[0].message.content or "")


def _client():
    import openai
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set — source the main-repo .env first")
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def cmd_judge(args):
    rows = [json.loads(l) for l in Path(args.samples).read_text().splitlines() if l.strip()]
    client = _client()
    verdicts = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_one, client, r["text"]): i for i, r in enumerate(rows)}
        for f in as_completed(futs):
            verdicts[futs[f]] = f.result()
    n = len(verdicts)
    rates = {k: sum(v[k] for v in verdicts) / n for k in INVARIANTS}
    print(json.dumps({"samples": args.samples, "n": n, "invariant_rates": rates}, indent=2))
    return rates


def cmd_suppression(args):
    normal = json.loads(Path(args.normal).read_text())["accuracy"]
    supp = json.loads(Path(args.suppressed).read_text())["accuracy"]
    out = {"normal_acc": normal, "suppressed_acc": supp, "shrinkage": normal - supp}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge"); j.add_argument("--samples", required=True)
    s = sub.add_parser("suppression")
    s.add_argument("--normal", required=True); s.add_argument("--suppressed", required=True)
    args = ap.parse_args()
    result = cmd_judge(args) if args.cmd == "judge" else cmd_suppression(args)
    Path("data/runs/tier5").mkdir(parents=True, exist_ok=True)
    p = Path("data/runs/tier5/attribution.json")
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev[args.cmd] = result
    p.write_text(json.dumps(prev, indent=2))
```

- [ ] **Step 5: Run the parse test to verify it passes**

Run: `python3 -m pytest tests/test_invariant_judge_parse.py -v`
Expected: PASS — 2 passed.

- [ ] **Step 6: Produce raw samples, judge usage, run suppression**

```bash
set -a; source /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.env; set +a
SFT=$(cat data/runs/tier5/sft_checkpoint.txt); TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"
# (a) raw SFT completions on a held-out family + a BBH set, then judge usage
.venv-tinker/bin/python scripts/tinker_eval.py --eval ledger \
  --corpus data/runs/tier5/eval_heldout_object_tracking.jsonl --model-path "$SFT" \
  --tokenizer $TOK --temperature 0 --max-tokens 4096 \
  --save-raw data/runs/tier5/raw_sft_heldout_tracking.jsonl \
  --out data/runs/tier5/t5_sft_heldout_tracking.json
python3 scripts/tier5_attribution.py judge --samples data/runs/tier5/raw_sft_heldout_tracking.jsonl
# (b) suppression: same eval with a forbidding system prompt, then compare accuracy
.venv-tinker/bin/python scripts/tinker_eval.py --eval ledger \
  --corpus data/runs/tier5/eval_heldout_object_tracking.jsonl --model-path "$SFT" \
  --tokenizer $TOK --temperature 0 --max-tokens 4096 \
  --system "Do NOT write out any explicit state, notes, tables, or a running ledger. Answer using only inline prose reasoning." \
  --out data/runs/tier5/t5_sftsupp_heldout_tracking.json
python3 scripts/tier5_attribution.py suppression \
  --normal data/runs/tier5/t5_sft_heldout_tracking.json \
  --suppressed data/runs/tier5/t5_sftsupp_heldout_tracking.json
```
Expected: `judge` prints per-invariant rates (`uses_ledger` high on transfer domains attributes uplift to the technique); `suppression` prints `shrinkage` > 0 (gains collapse toward base when state-tracking is forbidden). Both merged into `data/runs/tier5/attribution.json`.

- [ ] **Step 7: Commit**

```bash
git add scripts/tinker_eval.py scripts/tier5_attribution.py tests/test_invariant_judge_parse.py
git commit -m "feat(tier5): attribution — invariant-usage judge + suppression ablation"
```

---

## Task 12: Results writeup + mirror + research-log entry

Aggregate every eval into a results doc scored against the spec §7 thresholds with the claim-discipline line, mirror it to the main repo `docs/results/`, and append a research-log entry.

**Files:**
- Create: `docs/results/tier5_ledger_result.md`
- Create (mirror): `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/docs/results/tier5_ledger_result.md`
- Modify: `research_log.ipynb` (via the `/log` skill)

**Interfaces:**
- Consumes: `data/runs/tier5/gate_report.json`, all `data/runs/tier5/t5_*.json`, `data/runs/tier5/attribution.json`, the naturalization pass-rate printed in Task 6.
- Produces: the results doc + log entry (no code consumed downstream).

- [ ] **Step 1: Assemble the numbers**

Run:
```bash
python3 - <<'PY'
import json, glob
def acc(p):
    try: return round(json.load(open(p))["accuracy"], 3)
    except Exception: return None
rows = {}
for p in sorted(glob.glob("data/runs/tier5/t5_*.json")):
    rows[p.split("/")[-1][:-5]] = acc(p)
print(json.dumps(rows, indent=2))
PY
```
Expected: a dict of `t5_<tag>_<name> -> accuracy` for base/sft/rl across in-domain-extrap, held-out, 3 BBH sets, gsm8k control. Use these to fill the tables.

- [ ] **Step 2: Write `docs/results/tier5_ledger_result.md`**

Fill this template with the real numbers (base / SFT / SFT+RL per row):

```markdown
# Tier-5: Domain-General Inline Bookkeeping ("ledger") — Result

**Date:** 2026-07-02 · **Branch:** feature/activation-steering · **Model:** Qwen/Qwen3-30B-A3B-Instruct-2507

## Dataset
- Train traces: <N> ((prompt, completion)); 25% canonical / 75% Sonnet-naturalized; ~12% recovery; ~15% short no-ledger.
- Naturalization QC pass-rate: <PASS_RATE> (extract-back set-equality round-trip).

## Phase-1 gate (spec §5)
- Decision: <PROCEED|SKIP-RL|STOP>. mean_extrap_delta=<x>, mean_sft_acc=<y>. (from gate_report.json)

## §7.1 In-domain, extrapolated horizons (target ≥ +15pp where base collapses)
| set | base | SFT | SFT+RL | Δ(best−base) | pass? |
|---|---|---|---|---|---|
| trees d7 | | | | | |
| register N=150 | | | | | |
| graph scaled | | | | | |
| chain scaled | | | | | |

## §7.2 Held-out families zero-shot (target ≥ +10pp)
| family | base | SFT | SFT+RL | Δ | pass? |
|---|---|---|---|---|---|
| object_tracking | | | | | |
| scheduling | | | | | |

## §7.3 Real benchmarks (uplift on ≥2 of 3) + GSM8K control (degradation ≤ 2pp)
| benchmark | base | SFT | SFT+RL | Δ | pass? |
|---|---|---|---|---|---|
| BBH multistep_arithmetic_two | | | | | |
| BBH tracking_shuffled_objects | | | | | |
| BBH dyck_languages | | | | | |
| GSM8K (control) | | | | ≤2pp drop? | |

## §7.4 Attribution
- Invariant-usage rates on transfer domains (Sonnet judge): externalizes/grounds_steps/small_steps/restates/reads_off_answer/uses_ledger = <...>.
- Suppression ablation: normal_acc=<...>, suppressed_acc=<...>, shrinkage=<...>.

## Claim discipline (spec §7)
- "Generalizable technique" is claimed ONLY if §7.2 (held-out) AND §7.3 (benchmarks) both pass. §7.1 alone is an in-domain result. If SFT installed nothing in-domain (gate=STOP), the "in-context control is a capability property" negative is the reported result.
- **Verdict:** <one of: generalizable-technique confirmed | in-domain only | capability-property negative>.
```

- [ ] **Step 3: Mirror to the main repo**

Run:
```bash
cp docs/results/tier5_ledger_result.md \
   /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/docs/results/tier5_ledger_result.md
```
Expected: file present in both trees.

- [ ] **Step 4: Append a research-log entry**

Invoke the `/log` skill with a summary: the gate decision, the three §7 pass/fail verdicts, the naturalization QC pass-rate, and the final claim-discipline verdict. (The `/log` skill appends a timestamped entry to `research_log.ipynb`.)

- [ ] **Step 5: Commit**

```bash
git add docs/results/tier5_ledger_result.md research_log.ipynb
git commit -m "docs(tier5): ledger generalization result + research-log entry"
```

---

## Self-Review

**1. Spec coverage (§ by §):**
- §1 Goal/hypothesis (trainable domain-general bookkeeping; publishable negative) → Tasks 7–12; the gate STOP branch (Task 8) + claim-discipline (Task 12) capture the negative.
- §2 Decisions (inline ledger; 4 train / 2 held-out + ≥3 benchmarks; SFT→RL with eval gate; recovery traces; semantic-invariant naturalization; Qwen3-30B LoRA on Tinker) → Tasks 1–11 (form=inline ledger via ledger_protocol; families Tasks 2–4; SFT Task 7; gate Task 8; RL Task 9; recovery Task 1+6; naturalization Task 5; model in every Tinker command).
- §3 Protocol (five invariants; skeleton IR; canonical grammar ~25%; LLM naturalization ~75% with rejection-sampled round-trip QC) → Task 1 (IR + canonical grammar + verifier), Task 5 (naturalization + extract-back QC), Task 6 (25/75 split), Task 11 (five invariants encoded verbatim in the judge).
- §4 Task families (4 train + 2 held-out generators; horizon knobs; composition ~2–4k traces, 25/75, 15% no-ledger, 10–15% recovery) → Tasks 2–4 (generators with the exact keys/deps from the brief), Task 6 (composition: 2560 long, 12% recovery, 15% short, 25/75).
- §5 Phase-1 SFT + eval-checkpoint gate (proceed / skip-RL / stop) → Task 7 (SFT) + Task 8 (gate_report.json + three-way decision).
- §6 Phase-2 GRPO (cram boundary; reward answer_correct + λ·ledger_accuracy with λ=0.5, precision·recall spray-guard; canonical format anchor; deterministic paired parser) → Task 9 (`answer_correct + 0.5*ledger_reward`, `ledger_reward=P*R`, CANON_ANCHOR, parse_canonical) + Task 1 (spray-guard test).
- §7 Final eval (in-domain extrap ≥+15pp; held-out ≥+10pp; ≥3 benchmarks uplift ≥2/3 + GSM8K ≤2pp; attribution usage-judge + suppression; claim discipline) → Tasks 10–12.
- §8 New code list → ledger_protocol.py (T1), ledger_tasks/ (T2–4), naturalize_traces.py (T5), build_tier5_ledger.py (T6), SFT launcher (T7, satisfied by reusing tinker_sft.py per brief), tier5_evalsuite.sh (T10), GRPO launcher tinker_grpo_ledger.py (T9). All present.
- §9 Risks (capability-property → gate; format overfit → held-out + suppression; short-form regression → 15% no-ledger + GSM8K control; reward hacking → P·R; naturalization corruption → extract-back QC; billing → frequent checkpoints/save-every) → each has a corresponding mechanism in Tasks 6/8/9/11 and the Global Constraints.

**2. Placeholder scan:** No TBD/TODO/"similar to Task N". Every code step shows complete code; every run step shows an exact command + expected output. The only intentionally-templated content is the Task-12 results doc, whose blanks are experimental numbers that only exist after execution (each blank is named + sourced from a specific JSON), which is correct for a results artifact, not a plan gap.

**3. Type consistency across tasks:** `Note/Correction/Restate/Answer/Event/LedgerTask` and `render_canonical/parse_canonical/skeleton_claims/score_claims/inject_recovery` are defined once (Task 1) and used with identical signatures in Tasks 2–6, 9. Every generator exposes `gen(seed:int, horizon:int)->LedgerTask` (Tasks 2–4) and is reached only via `get_generator` (Task 2), used in Tasks 5, 6. `score_claims` returns exactly `{precision, recall, f1, answer_correct, ledger_reward}`; `ledger_reward = precision*recall` is consumed identically by the Task-9 env and the Task-1/Task-9 reward tests. Eval rows carry `answer` (Task 6) consumed by the `ledger` scorer (Task 8) and the suite (Task 10). RL-pool rows carry `{prompt, gold_answer, gold_facts, family, horizon}` (Task 6) consumed by `LedgerEnv`/`LedgerDatasetBuilder` (Task 9). Checkpoint sidecars `data/runs/tier5/{sft,rl}_checkpoint.txt` are written in Tasks 7/9 and read in Tasks 8/10/11. The judge model id `anthropic/claude-sonnet-4-6` is identical in Tasks 5 and 11.

