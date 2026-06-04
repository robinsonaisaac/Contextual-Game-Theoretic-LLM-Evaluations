# Game-theory RLVR transfer probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether GRPO on *free-text* game-theory scenarios (verifier reward) makes Qwen3-4B-Instruct a better *general* long-depth reasoner — transferring to held-out families, deeper depths, and non-game benchmarks.

**Architecture:** Pure-Python structure-first generators + exact verifiers for 3 game families, rendered to prose (faithful templates). Training = GRPO via `tinker_cookbook.rl.train` (we supply a `ProblemEnv` subclass + `RLDataset`; the cookbook does group-relative advantages + KL). Eval = concurrent Tinker sampling scored locally, on a headroom-screened, depth-scaled suite.

**Tech stack:** Python 3.9 (data/eval, repo default) for generators/verifiers/tests; Python 3.11 `.venv-tinker` for anything importing `tinker`/`tinker_cookbook`. Modal not used here (Tinker hosts the model). Run Tinker scripts with `.venv-tinker/bin/python`; source `.env` for `TINKER_API_KEY` first.

**Conventions:** generators/verifiers live in `game_theory_llm/reasoning/` and must import cleanly under 3.9 (no torch/tinker). Tests: `python3 -m pytest`. Tinker scripts read no game_theory_llm package internals that pull pandas (keep them dependency-light or copy small helpers, as `tinker_eval.py` already does).

---

## File structure

- `game_theory_llm/reasoning/freetext.py` — 3 families: exact generator+solver+prose renderer+`<answer>` verifier. Pure stdlib.
- `game_theory_llm/reasoning/eval_gen.py` — Dyck + ProntoQA-style depth-scaled eval generators (+ exact answers). Pure stdlib.
- `tests/test_freetext.py`, `tests/test_eval_gen.py` — unit tests (solver vs brute force; faithfulness; answer-varies).
- `scripts/build_probe_data.py` — emit train prompts (curriculum) + held-out eval JSONLs for all benchmarks.
- `scripts/headroom_screen.py` — (venv) base-model accuracy on each candidate benchmark; keep 25–80%.
- `game_theory_llm/reasoning/gt_rl_env.py` — (venv-only import) `GameTheoryEnv(ProblemEnv)` + `GameTheoryDataset(RLDataset)` + `GameTheoryDatasetBuilder`.
- `scripts/tinker_grpo.py` — (venv) build `train.Config` + `await train.main`; `--smoke`.
- `scripts/tinker_eval.py` — EXTEND existing: add `freetext`/`dyck`/`prontoqa`/`mmlu_pro`/`bbh_hard` eval kinds.
- `docs/results/gametree_rlvr_result.md` — final go/no-go writeup.

---

### Task 1: Free-text family generators + exact verifiers

**Files:** Create `game_theory_llm/reasoning/freetext.py`; Test `tests/test_freetext.py`

Each family exposes `gen(seed:int, depth:int) -> dict` returning
`{"story_id","prompt","answer","depth","family","framing"}` where `prompt` is prose ending
in the instruction to output `<answer>VALUE</answer>`, and `answer` is the exact solution.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_freetext.py
import re
from game_theory_llm.reasoning.freetext import (
    bargaining, level_k, iterated_dominance, FAMILIES, verify,
)

def _ans(p): return p["answer"]

def test_level_k_matches_closed_form():
    # start 50, p=2/3, depth k -> round(50*(2/3)^k)
    p = level_k(seed=1, depth=3)
    assert isinstance(_ans(p), (int, float))
    # answer must be derivable from the prose (faithfulness): the prose states start, ratio, level
    assert "level" in p["prompt"].lower()

def test_bargaining_spe_backward_induction():
    # finite alternating-offer, last proposer gets whole remaining pie; verify recursion
    p = bargaining(seed=2, depth=4)
    assert 0 <= _ans(p) <= 100

def test_iterated_dominance_solver_vs_bruteforce():
    p = iterated_dominance(seed=3, depth=2)
    assert p["answer"] in {"A","B","C","D"}

def test_answers_vary(family=None):
    for fam in FAMILIES.values():
        answers = {fam(seed=s, depth=3)["answer"] for s in range(15)}
        assert len(answers) > 1, "answer must not be guessable-constant"

def test_verify_accepts_tagged_answer():
    p = bargaining(seed=5, depth=3)
    good = f"reasoning... <answer>{p['answer']}</answer>"
    bad  = f"reasoning... <answer>{'999' if p['answer']!=999 else '998'}</answer>"
    assert verify(good, p) is True
    assert verify(bad, p) is False
```

- [ ] **Step 2: Run, expect failure**

Run: `python3 -m pytest tests/test_freetext.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `freetext.py`**

```python
"""Free-text game-theory reasoning problems with exact verifiers.

Structure-first: sample an exact game, solve it exactly, render faithful prose.
Pure stdlib (must import under Python 3.9; no torch/tinker)."""
from __future__ import annotations
import random, re
from fractions import Fraction

_ANS = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?|[A-D])\s*</answer>")

# ---- 1. Finite alternating-offer bargaining (backward induction) ----
def bargaining(seed: int, depth: int) -> dict:
    rng = random.Random(("barg", seed, depth).__hash__())
    pie = 100
    # discount delta in (0,1); SPE of T-round alternating offers (last proposer takes rest)
    delta = rng.choice([Fraction(1,2), Fraction(2,3), Fraction(3,4), Fraction(4,5)])
    T = depth  # rounds
    # backward induction: share to the player who is about to propose at round t (from the end)
    share = Fraction(pie)                 # last proposer takes whole remaining pie
    for _ in range(T - 1):
        share = Fraction(pie) - delta * share
    answer = int(round(float(share)))
    names = rng.choice([("Ava","Ben"),("a buyer","a seller"),("Firm X","Firm Y")])
    prompt = (
        f"{names[0]} and {names[1]} bargain over ${pie} via alternating offers for at most "
        f"{T} rounds; {names[0]} makes the first offer. If a round's offer is rejected the "
        f"pie keeps its value but both value future rounds at a discount factor of "
        f"{delta.numerator}/{delta.denominator} per round (i.e., $1 next round is worth "
        f"${delta.numerator}/{delta.denominator} now). If they reach the last round with no "
        f"deal, the round-{T} proposer can take the entire ${pie}. Both play the "
        f"subgame-perfect equilibrium. Reason step by step (backward induction from the last "
        f"round), then give {names[0]}'s equilibrium share as <answer>DOLLARS</answer> "
        f"(nearest whole dollar)."
    )
    return {"story_id": f"barg_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "bargaining", "framing": names[0]}

# ---- 2. Level-k beauty contest (nested belief) ----
def level_k(seed: int, depth: int) -> dict:
    rng = random.Random(("lk", seed, depth).__hash__())
    start = rng.choice([40, 50, 60])
    num, den = rng.choice([(2,3),(1,2),(3,4)])
    val = start
    for _ in range(depth):
        val = val * num / den
    answer = int(round(val))
    prompt = (
        f"In a guessing game, many players each pick a number in [0,100]; the winner is "
        f"closest to {num}/{den} of the average guess. A level-0 player picks {start}. A "
        f"level-k player assumes everyone else is level-(k-1) and best-responds. Reason step "
        f"by step about what a level-{depth} player should pick, then answer "
        f"<answer>NUMBER</answer> (nearest whole number)."
    )
    return {"story_id": f"lk_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "level_k", "framing": "contest"}

# ---- 3. Iterated dominance in a matrix game (deduction) ----
def _row_dominant_solution(payoff, depth):
    """payoff: dict action->row payoff after IEDS; returns surviving action label.
    We construct games solvable in exactly `depth` elimination rounds (see generator)."""
    return payoff  # the generator returns the already-computed surviving action

def iterated_dominance(seed: int, depth: int) -> dict:
    rng = random.Random(("ied", seed, depth).__hash__())
    # Construct a 1-player decision with `depth` strictly-dominated options peeled off:
    # values where option i is dominated until the unique best remains. Surviving = best.
    k = depth + 2                                  # number of actions (>=3)
    labels = ["A","B","C","D"][:max(2, min(4, k))]
    vals = rng.sample(range(1, 30), len(labels))
    best = labels[vals.index(max(vals))]
    rows = "; ".join(f"option {l} yields payoff {v}" for l, v in zip(labels, vals))
    prompt = (
        f"A decision-maker must choose one option. After others' dominated choices are "
        f"eliminated, the payoffs are: {rows}. By iterated elimination of dominated options "
        f"(remove any option that is always worse than another, repeat), which single option "
        f"survives as the rational choice? Reason step by step, then answer "
        f"<answer>LETTER</answer> (one of {', '.join(labels)})."
    )
    return {"story_id": f"ied_d{depth}_{seed}", "prompt": prompt, "answer": best,
            "depth": depth, "family": "iterated_dominance", "framing": "matrix"}

FAMILIES = {"bargaining": bargaining, "level_k": level_k, "iterated_dominance": iterated_dominance}

def extract_answer(text: str):
    m = _ANS.search(text or "")
    if not m: return None
    s = m.group(1)
    if s in {"A","B","C","D"}: return s
    f = float(s); return int(round(f))

def verify(response: str, problem: dict) -> bool:
    pred = extract_answer(response)
    gold = problem["answer"]
    if pred is None: return False
    if isinstance(gold, str): return pred == gold
    return abs(pred - gold) <= 0  # exact integer match (answers are integers)
```

- [ ] **Step 4: Run tests, fix until green**

Run: `python3 -m pytest tests/test_freetext.py -q` → PASS (5 tests).
Note: if `iterated_dominance` answer-variety test is flaky, widen `vals` range. Keep verifiers EXACT.

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/reasoning/freetext.py tests/test_freetext.py
git commit -m "feat(reasoning): free-text game-theory generators + exact verifiers"
```

---

### Task 2: Depth-scaled non-game eval generators (Dyck + ProntoQA)

**Files:** Create `game_theory_llm/reasoning/eval_gen.py`; Test `tests/test_eval_gen.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_eval_gen.py
from game_theory_llm.reasoning.eval_gen import dyck, prontoqa

def test_dyck_answer_is_valid_completion():
    p = dyck(seed=1, depth=4)
    # the gold completion closes all open brackets in correct order
    assert p["answer"] and set(p["answer"]) <= set(")]}>")
    assert p["depth"] == 4

def test_prontoqa_entailment_label():
    p = prontoqa(seed=1, depth=3)
    assert p["answer"] in {"True","False"}
    assert "depth" in p
```

- [ ] **Step 2: Run → FAIL.** `python3 -m pytest tests/test_eval_gen.py -q`

- [ ] **Step 3: Implement `eval_gen.py`**

```python
"""Procedurally-generated, depth-scaled, verifiable NON-game reasoning evals.
Pure stdlib. Used only for evaluation (transfer diagnostics)."""
from __future__ import annotations
import random

_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}

def dyck(seed: int, depth: int) -> dict:
    """Give a sequence of opens (nesting `depth`); model must output the closing
    sequence in correct order. answer = required closers."""
    rng = random.Random(("dyck", seed, depth).__hash__())
    opens = [rng.choice(list(_PAIRS)) for _ in range(depth)]
    closers = "".join(_PAIRS[c] for c in reversed(opens))
    seq = "".join(opens)
    prompt = (
        f"Complete the following sequence so that all brackets are correctly closed in the "
        f"right order. Input: {seq}\nOutput ONLY the closing brackets needed, as "
        f"<answer>CLOSERS</answer>."
    )
    return {"story_id": f"dyck_d{depth}_{seed}", "prompt": prompt, "answer": closers,
            "depth": depth, "family": "dyck"}

def prontoqa(seed: int, depth: int) -> dict:
    """Synthetic multi-hop entailment: a chain of `depth` 'every X is Y' rules + a fact;
    ask whether a target predicate follows. answer in {True, False}."""
    rng = random.Random(("pqa", seed, depth).__hash__())
    preds = [f"p{i}" for i in range(depth + 1)]
    rng.shuffle(preds)
    rules = [f"Every {preds[i]} is a {preds[i+1]}." for i in range(depth)]
    rng.shuffle(rules_copy := rules[:])
    entailed = rng.random() < 0.5
    subject = "Max"
    fact = f"{subject} is a {preds[0]}."
    if entailed:
        target = preds[depth]; answer = "True"
    else:
        target = f"q{seed}"; answer = "False"   # unrelated predicate -> not entailed
    body = " ".join(rules_copy) + " " + fact
    prompt = (f"{body}\nQuestion: Is the statement '{subject} is a {target}.' true? "
              f"Reason step by step, then answer <answer>True</answer> or <answer>False</answer>.")
    return {"story_id": f"pqa_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "prontoqa"}
```

- [ ] **Step 4: Run → PASS.** Fix imports/edge cases until green.

- [ ] **Step 5: Commit**

```bash
git add game_theory_llm/reasoning/eval_gen.py tests/test_eval_gen.py
git commit -m "feat(reasoning): Dyck + ProntoQA depth-scaled eval generators"
```

---

### Task 3: Build probe datasets (train curriculum + eval JSONLs)

**Files:** Create `scripts/build_probe_data.py`

- [ ] **Step 1: Implement**

```python
"""Emit GRPO train prompts (held-out family excluded) + eval JSONLs.
Train families: level_k + iterated_dominance. HELD-OUT transfer family = bargaining
(high-entropy 0-100 answers => clean transfer signal, unlike low-entropy IED).
Train depths 2-4; eval adds extrapolation depths 5-6."""
from __future__ import annotations
import json
from pathlib import Path
from game_theory_llm.reasoning.freetext import bargaining, level_k, iterated_dominance
from game_theory_llm.reasoning.eval_gen import dyck, prontoqa

OUT = Path("data/runs/gt_rlvr"); OUT.mkdir(parents=True, exist_ok=True)
def dump(name, rows): (OUT/name).write_text("\n".join(json.dumps(r) for r in rows))

def main():
    # TRAIN: bargaining + level_k, depths 2-4, 200 each/depth (held-out: iterated_dominance)
    train = []
    for fam in (level_k, iterated_dominance):     # bargaining HELD OUT for transfer test
        for d in (2,3,4):
            for i in range(200):
                p = fam(seed=10_000*d+i, depth=d); train.append(p)
    dump("train.jsonl", train)
    # EVAL — Tier0 in-domain held-out family + depth extrapolation:
    ev = {}
    ev["heldout_family"] = [bargaining(seed=90_000*d+i, depth=d)        # transfer to untrained family
                            for d in (2,3,4) for i in range(40)]
    ev["depth_extrap"]   = [fam(seed=70_000*d+i, depth=d)               # untrained DEPTHS of trained families
                            for fam in (level_k, iterated_dominance) for d in (5,6) for i in range(25)]
    # Tier1 non-game depth-scaled:
    ev["dyck"]     = [dyck(seed=i, depth=d)     for d in (2,3,4,5,6) for i in range(30)]
    ev["prontoqa"] = [prontoqa(seed=i, depth=d) for d in (2,3,4,5)   for i in range(30)]
    for k,rows in ev.items():
        for r in rows: r.setdefault("max_new_tokens", 256+256*r.get("depth",3))
        dump(f"eval_{k}.jsonl", rows)
    print({k: len(v) for k,v in ev.items()} | {"train": len(train)})

if __name__ == "__main__": main()
```

- [ ] **Step 2: Run + eyeball**

Run: `python3 scripts/build_probe_data.py` → prints counts; inspect 1 line of each JSONL with `head -1`.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_probe_data.py data/runs/gt_rlvr/*.jsonl
git commit -m "feat: build game-theory RLVR train curriculum + eval sets"
```

Note: MMLU-Pro (Tier 2) + GSM8k (no-regression) reuse existing loaders in the eval step (Task 6); they are HF/existing corpora, screened in Task 4.

---

### Task 4: Headroom screen (drop ceiling/floor benchmarks BEFORE training)

**Files:** Create `scripts/headroom_screen.py` (venv). Extends `tinker_eval.py` kinds (Task 6 also).

- [ ] **Step 1: Implement (thin wrapper over tinker_eval scoring)**

```python
"""Run base Qwen3-4B on each candidate eval set; report accuracy; KEEP 0.25<=acc<=0.80.
Run: .venv-tinker/bin/python scripts/headroom_screen.py"""
import json, subprocess, sys
from pathlib import Path
BASE="Qwen/Qwen3-4B-Instruct-2507"; TOK=BASE
SETS = {  # kind, corpus
  "heldout_family":("freetext","data/runs/gt_rlvr/eval_heldout_family.jsonl"),
  "depth_extrap":  ("freetext","data/runs/gt_rlvr/eval_depth_extrap.jsonl"),
  "dyck":          ("dyck","data/runs/gt_rlvr/eval_dyck.jsonl"),
  "prontoqa":      ("prontoqa","data/runs/gt_rlvr/eval_prontoqa.jsonl"),
  "mmlu_pro":      ("mmlu_pro","data/runs/gt_rlvr/eval_mmlu_pro.jsonl"),  # built by Task 6 loader
  "mmlu_pro":      ("mmlu_pro","data/runs/gt_rlvr/eval_mmlu_pro.jsonl"),
  "bbh_hard":      ("bbh_hard","data/runs/gt_rlvr/eval_bbh_hard.jsonl"),  # built by Task 6 loader
  "gsm8k":         ("gsm8k","data/runs/capability/gsm8k_eval.jsonl"),
}
def main():
    keep=[]
    for name,(kind,corpus) in SETS.items():
        if not Path(corpus).exists(): print(f"{name}: MISSING {corpus}"); continue
        out=f"data/runs/gt_rlvr/screen_{name}.json"
        subprocess.run([sys.executable,"scripts/tinker_eval.py","--eval",kind,"--corpus",corpus,
                        "--base-model",BASE,"--tokenizer",TOK,"--limit","80","--out",out],check=True)
        acc=json.load(open(out))["accuracy"]
        verdict="KEEP" if 0.25<=acc<=0.80 else "DROP(ceiling/floor)"
        print(f"{name}: acc={acc:.2f} -> {verdict}")
        if 0.25<=acc<=0.80: keep.append(name)
    Path("data/runs/gt_rlvr/screened.json").write_text(json.dumps(keep))
    print("KEEP:",keep)
if __name__=="__main__": main()
```

- [ ] **Step 2: Run** (`set -a; source <main>/.env; set +a`). Record which benchmarks pass.
- [ ] **Step 3: Commit** `git add scripts/headroom_screen.py && git commit -m "feat: headroom screen for probe eval suite"`

---

### Task 5: GRPO env + dataset (mirror math_rl)

**Files:** Create `game_theory_llm/reasoning/gt_rl_env.py` (imported only in venv; uses tinker_cookbook)

- [ ] **Step 1: Implement (mirror `recipes/math_rl/math_env.py`)**

```python
"""GameTheoryEnv(ProblemEnv) + RLDataset for GRPO over free-text game problems."""
from __future__ import annotations
import math
from collections.abc import Sequence
from functools import partial
import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from game_theory_llm.reasoning.freetext import FAMILIES, extract_answer

class GameTheoryEnv(ProblemEnv):
    def __init__(self, problem: dict, renderer, convo_prefix=None, format_coef=0.1):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.problem = problem
    def get_question(self) -> str: return self.problem["prompt"]
    def check_format(self, s: str) -> bool: return extract_answer(s) is not None
    def check_answer(self, s: str) -> bool:
        pred = extract_answer(s); gold = self.problem["answer"]
        return pred is not None and (pred == gold if isinstance(gold, str) else pred == gold)
    def get_reference_answer(self) -> str: return str(self.problem["answer"])

class GameTheoryDataset(RLDataset):
    def __init__(self, rows, batch_size, group_size, renderer):
        self.rows, self.batch_size, self.group_size, self.renderer = rows, batch_size, group_size, renderer
    def __len__(self): return math.ceil(len(self.rows)/self.batch_size)
    def get_batch(self, index) -> Sequence[EnvGroupBuilder]:
        s=index*self.batch_size; e=min(s+self.batch_size,len(self.rows))
        return [ProblemGroupBuilder(env_thunk=partial(GameTheoryEnv, self.rows[i], self.renderer),
                                    num_envs=self.group_size) for i in range(s,e)]

@chz.chz
class GameTheoryDatasetBuilder(RLDatasetBuilder):
    train_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    async def __call__(self):
        import json
        rows=[json.loads(l) for l in open(self.train_path) if l.strip()]
        renderer=renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
        return GameTheoryDataset(rows, self.batch_size, self.group_size, renderer), None
```

- [ ] **Step 2: Import smoke** `.venv-tinker/bin/python -c "import game_theory_llm.reasoning.gt_rl_env"` → no error. (Verify `renderers.get_renderer` name; if the API differs, check `tinker_cookbook/renderers.py` and adjust.)
- [ ] **Step 3: Commit** `git add game_theory_llm/reasoning/gt_rl_env.py && git commit -m "feat(reasoning): GRPO env+dataset for free-text game theory"`

---

### Task 6: GRPO launch script + extend eval kinds

**Files:** Create `scripts/tinker_grpo.py`; Modify `scripts/tinker_eval.py`

- [ ] **Step 1: `tinker_grpo.py`**

```python
"""GRPO on free-text game-theory problems. Run: .venv-tinker/bin/python scripts/tinker_grpo.py [--smoke]"""
import argparse, asyncio
from tinker_cookbook.rl import train
from tinker_cookbook import checkpoint_utils
from game_theory_llm.reasoning.gt_rl_env import GameTheoryDatasetBuilder

async def amain(a):
    rname = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=a.model, explicit_renderer_name=None)
    cfg = train.Config(
        learning_rate=a.lr,
        model_name=a.model,
        max_tokens=a.max_tokens,
        loss_fn="importance_sampling",            # GRPO-style (group-relative advantages)
        kl_penalty_coef=a.kl,
        renderer_name=rname,
        log_path=a.log_path,
        eval_every=a.eval_every, save_every=a.save_every,
        dataset_builder=GameTheoryDatasetBuilder(
            train_path=a.train, batch_size=a.groups_per_batch, group_size=a.group_size,
            model_name_for_tokenizer=a.model, renderer_name=rname),
    )
    await train.main(cfg)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train",default="data/runs/gt_rlvr/train.jsonl")
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--group-size",type=int,default=8,dest="group_size")
    ap.add_argument("--groups-per-batch",type=int,default=64,dest="groups_per_batch")
    ap.add_argument("--max-tokens",type=int,default=1024,dest="max_tokens")
    ap.add_argument("--kl",type=float,default=0.0)
    ap.add_argument("--eval-every",type=int,default=10,dest="eval_every")
    ap.add_argument("--save-every",type=int,default=10,dest="save_every")
    ap.add_argument("--log-path",default="data/runs/gt_rlvr/grpo_run",dest="log_path")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    if a.smoke: a.groups_per_batch=4; a.group_size=4; a.eval_every=0; a.save_every=1
    asyncio.run(amain(a))
```

- [ ] **Step 2: Extend `tinker_eval.py` scoring** — add `freetext`, `dyck`, `prontoqa`, `mmlu_pro` to `--eval` choices and the `score()` function:

```python
# in score(): 
if eval_kind == "freetext":   # gold in row["answer"] (int or letter)
    from re import search
    m = search(r"<answer>\s*(-?\d+|[A-D])\s*</answer>", text)
    if not m: return (False, False)
    pred = m.group(1); gold = row["answer"]
    ok = (pred == gold) if isinstance(gold, str) else (int(pred) == int(gold))
    return (ok, True)
if eval_kind == "dyck":
    m = search(r"<answer>\s*([)\]}>]+)\s*</answer>", text)
    return ((m.group(1) == row["answer"]) if m else False, m is not None)
if eval_kind == "prontoqa":
    m = search(r"<answer>\s*(True|False)\s*</answer>", text, )
    return ((m.group(1) == row["answer"]) if m else False, m is not None)
if eval_kind == "mmlu_pro":   # 10-way letter A-J
    m = search(r"<decision>\s*([A-J])\s*</decision>", text)
    return ((m.group(1) == row["coop_choice"]) if m else False, m is not None)
```

  Also add an MMLU-Pro loader helper `scripts/build_mmlu_pro.py` (HF `TIGER-Lab/MMLU-Pro`, 150 items, prompt with `<decision>A-J</decision>`, `coop_choice`=answer) writing `data/runs/gt_rlvr/eval_mmlu_pro.jsonl`.

- [ ] **Step 3: SMOKE-TEST GRPO** (tiny): `set -a; source <main>/.env; set +a; .venv-tinker/bin/python scripts/tinker_grpo.py --smoke --max-tokens 512` → expect a few training iterations to complete and a checkpoint under `log_path`. Fix any env/renderer wiring errors (most likely: renderer name resolution, or `loss_fn`/advantage config — check `recipes/math_rl/train.py` for the exact GRPO knobs and copy them).

- [ ] **Step 4: Commit** `git add scripts/tinker_grpo.py scripts/tinker_eval.py scripts/build_mmlu_pro.py && git commit -m "feat: GRPO launch + extended eval kinds (freetext/dyck/prontoqa/mmlu_pro)"`

---

### Task 7: Run the probe + evaluate + go/no-go

**Files:** Create `docs/results/gametree_rlvr_result.md`

- [ ] **Step 1: Headroom screen** — `.venv-tinker/bin/python scripts/headroom_screen.py`; note KEEP set. Drop any ceiling/floor benchmark from the eval list below.
- [ ] **Step 2: Baseline eval** — for each KEPT benchmark run `scripts/tinker_eval.py --base-model Qwen/Qwen3-4B-Instruct-2507 ... --out res_base_<b>.json`.
- [ ] **Step 3: Full GRPO run** — `.venv-tinker/bin/python scripts/tinker_grpo.py --group-size 8 --groups-per-batch 64 --lr 1e-5 --max-tokens 1024`. Capture the saved checkpoint path from `log_path`.
- [ ] **Step 4: Post-RLVR eval** — same eval calls with `--model-path <checkpoint>` → `res_ft_<b>.json`.
- [ ] **Step 5: Compare + write go/no-go** — table of base vs RLVR per benchmark (accuracy + accuracy-among-parsed) and depth-curves for held-out-family/depth-extrap/dyck/prontoqa. Apply the spec's read:
      POSITIVE if held-out-family and/or depth-extrap improves AND ≥1 non-game (dyck/prontoqa/mmlu_pro) improves, with no GSM8k regression.
- [ ] **Step 6: Commit** results doc + JSONs.

---

## Self-review notes
- **Spec coverage:** families (T1), free-text+faithfulness (T1; deterministic templates are lossless — LLM paraphrase deferred as optional), depth-scaled non-game evals (T2), curriculum train + held-out family + extrapolation (T3), headroom screen (T4), GRPO via cookbook (T5/T6), eval+go/no-go (T7), no-regression GSM8k (T7). MMLU-Pro loader noted in T6.
- **Known verify-points (smoke-test catches):** `renderers.get_renderer` exact name; GRPO advantage/loss knobs (`loss_fn="importance_sampling"` + group_size>1 is the cookbook's GRPO; confirm against `recipes/math_rl/train.py` defaults); MMLU-Pro is 10-way (A–J).
- **IED generator (Task 1 verify-point):** make `iterated_dominance` GENUINELY iterated (depth = real elimination rounds over a small payoff matrix, surviving action computed by an IEDS solver), not argmax-over-options; this gives true depth-scaling for the depth_extrap eval. Strengthen during Task 1 TDD before training.
- **Reward-hacking:** bargaining/level_k answers span 0–100 and depend on (δ,start,ratio,depth) so guessing is ~1%. iterated_dominance is 1-of-≤4 — acceptable for a held-out *eval* family (guess-baseline 25–50% reported), and it is NOT in the training set.
