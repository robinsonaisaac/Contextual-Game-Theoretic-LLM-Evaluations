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
