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
