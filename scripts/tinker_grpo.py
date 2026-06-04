"""GRPO on free-text game-theory problems via the Tinker cookbook rl.train loop.

RUN WITH:  .venv-tinker/bin/python scripts/tinker_grpo.py [--smoke]
(source the main repo's .env first for TINKER_API_KEY)

The cookbook does GRPO (group-relative advantages from --group-size rollouts/prompt,
KL control, optim). We only supply GameTheoryDatasetBuilder (prompt + verifier reward).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the standalone env module
from gt_rl_env import GameTheoryDatasetBuilder  # noqa: E402

from tinker_cookbook import checkpoint_utils  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402


async def amain(a):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=a.model, explicit_renderer_name=None, load_checkpoint_path=None
    )
    kl_ref = train.KLReferenceConfig(base_model=a.model) if a.kl > 0 else None
    config = train.Config(
        learning_rate=a.lr,
        model_name=a.model,
        max_tokens=a.max_tokens,
        kl_penalty_coef=a.kl,
        kl_reference_config=kl_ref,
        renderer_name=renderer_name,
        log_path=a.log_path,
        eval_every=a.eval_every,
        save_every=a.save_every,
        dataset_builder=GameTheoryDatasetBuilder(
            train_path=a.train,
            batch_size=a.groups_per_batch,
            group_size=a.group_size,
            model_name_for_tokenizer=a.model,
            renderer_name=renderer_name,
        ),
    )
    await train.main(config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train", default="data/runs/gt_rlvr/train.jsonl")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group-size", type=int, default=8, dest="group_size")
    ap.add_argument("--groups-per-batch", type=int, default=64, dest="groups_per_batch")
    ap.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=10, dest="save_every")
    ap.add_argument("--log-path", default="data/runs/gt_rlvr/grpo_run", dest="log_path")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.groups_per_batch, a.group_size = 4, 4
        a.eval_every, a.save_every, a.max_tokens = 0, 2, 512
        a.train = "data/runs/gt_rlvr/train.jsonl"
    asyncio.run(amain(a))
