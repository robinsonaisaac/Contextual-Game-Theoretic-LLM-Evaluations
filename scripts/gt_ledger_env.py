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
