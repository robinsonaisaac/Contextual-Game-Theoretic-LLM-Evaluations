"""Step-3 RL env: multi-round stateless-with-memory episodes (trains the MEMORY STRATEGY).

Each round the policy sees ONLY (problem + saved notes table); reasoning text does not
survive rounds. The policy writes `NOTE: key = value` lines and ends with CONTINUE or
<answer>N</answer>. Reward (end of episode) = 1{answer correct} − 0.1·1{no answer given}.

Unlike Tier-1..3 (which could only tune slip rates on an algorithm the model already had),
the trainable object here is the note-taking/decomposition strategy — a policy-level
behavior with a plausible transfer mechanism (memory discipline is task-general).

Imported ONLY in the Tinker 3.11 venv. Mirrors scripts/gt_rl_env.py conventions.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from functools import partial

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action, ActionExtra, Env, EnvGroupBuilder, Observation, RLDataset,
    RLDatasetBuilder, StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

NOTE = re.compile(r"^\s*NOTE:\s*(\S{1,40})\s*=\s*(.{1,40}?)\s*$", re.MULTILINE)
ANS_INT = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
MAX_NOTES = 300

INSTR = """
=== MULTI-ROUND MODE WITH EXTERNAL MEMORY ===
You are solving this in up to {R} rounds. IMPORTANT: your reasoning text is NOT carried
between rounds — only your saved notes survive. Anything you do not save is lost.

To save a fact, output a line of exactly this form (one per line, integer or short value):
NOTE: <key> = <value>
For example, to record that the subtree reached by moves A then B is worth -3:
NOTE: A.B = -3

Work on a manageable chunk this round, save what you conclude as notes, then end your
reply with the single word CONTINUE. In a later round, combine saved notes to finish.
When you know the final answer, output it as <answer>NUMBER</answer> (this ends the task).

This is round {r} of {R}. Your saved notes so far:
{table}
"""


class MemoryGameEnv(Env):
    def __init__(self, problem: dict, renderer, rounds: int = 6):
        self.problem = problem
        self.renderer = renderer
        self.rounds = rounds
        self.r = 0
        self.notes: dict = {}

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _obs(self) -> Observation:
        table = "\n".join(f"  {k} = {v}" for k, v in self.notes.items()) or "  (none yet)"
        prompt = self.problem["prompt"] + INSTR.format(R=self.rounds, r=self.r, table=table)
        return self.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}])

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self.r = 1
        return self._obs(), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _term = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        for k, v in NOTE.findall(content or ""):
            if len(self.notes) < MAX_NOTES or k in self.notes:
                self.notes[k] = v
        m = ANS_INT.search(content or "")
        done = m is not None or self.r >= self.rounds
        if not done:
            self.r += 1
            return StepResult(reward=0.0, episode_done=False,
                              next_observation=self._obs(),
                              next_stop_condition=self.stop_condition,
                              metrics={})
        correct = float(m is not None and int(m.group(1)) == int(self.problem["answer"]))
        reward = correct - (0.1 if m is None else 0.0)
        return StepResult(reward=reward, episode_done=True,
                          next_observation=tinker.ModelInput.empty(),
                          next_stop_condition=self.stop_condition,
                          metrics={"correct": correct, "format": float(m is not None),
                                   "rounds_used": float(self.r),
                                   "n_notes": float(len(self.notes))})


class MemoryGameDataset(RLDataset):
    def __init__(self, rows, batch_size, group_size, renderer, rounds):
        self.rows, self.batch_size, self.group_size = rows, batch_size, group_size
        self.renderer, self.rounds = renderer, rounds

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        return [ProblemGroupBuilder(
                    env_thunk=partial(MemoryGameEnv, self.rows[i], self.renderer,
                                      self.rounds),
                    num_envs=self.group_size)
                for i in range(start, end)]


@chz.chz
class MemoryDatasetBuilder(RLDatasetBuilder):
    train_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    rounds: int = 6

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        rows = [json.loads(l) for l in open(self.train_path) if l.strip()]
        renderer = renderers.get_renderer(
            self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
        return MemoryGameDataset(rows, self.batch_size, self.group_size, renderer,
                                 self.rounds), None
