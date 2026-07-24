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
