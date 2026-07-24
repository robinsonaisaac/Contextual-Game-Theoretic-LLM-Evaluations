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
