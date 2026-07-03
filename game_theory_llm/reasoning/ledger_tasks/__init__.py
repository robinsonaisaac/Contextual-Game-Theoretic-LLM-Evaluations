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
