"""GRPO env + dataset for free-text game-theory RLVR (mirrors recipes/math_rl).

Imported ONLY in the Tinker 3.11 venv (depends on tinker_cookbook). The cookbook's
rl.train loop does GRPO (group-relative advantages from `num_envs` rollouts/prompt,
KL control, optim); we only supply the question + exact verifier reward.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from functools import partial

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Inlined from game_theory_llm.reasoning.freetext to avoid importing the heavy
# game_theory_llm package (scipy/pandas) in the lightweight Tinker venv.
_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def extract_answer(text: str):
    m = _ANS.search(text or "")
    return int(m.group(1)) if m else None


class GameTheoryEnv(ProblemEnv):
    def __init__(self, problem: dict, renderer, convo_prefix=None, format_coef: float = 0.1):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.problem = problem

    def get_question(self) -> str:
        return self.problem["prompt"]

    def check_format(self, sample_str: str) -> bool:
        return extract_answer(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        pred = extract_answer(sample_str)
        return pred is not None and pred == self.problem["answer"]

    def get_reference_answer(self) -> str:
        return str(self.problem["answer"])


class GameTheoryDataset(RLDataset):
    def __init__(self, rows: list[dict], batch_size: int, group_size: int, renderer):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        return [
            ProblemGroupBuilder(
                env_thunk=partial(GameTheoryEnv, self.rows[i], self.renderer),
                num_envs=self.group_size,
            )
            for i in range(start, end)
        ]


@chz.chz
class GameTheoryDatasetBuilder(RLDatasetBuilder):
    train_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        rows = [json.loads(l) for l in open(self.train_path) if l.strip()]
        renderer = renderers.get_renderer(
            self.renderer_name, get_tokenizer(self.model_name_for_tokenizer)
        )
        return GameTheoryDataset(rows, self.batch_size, self.group_size, renderer), None
