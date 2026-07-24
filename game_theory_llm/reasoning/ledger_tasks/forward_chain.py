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
