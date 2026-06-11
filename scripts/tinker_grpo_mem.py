"""GRPO with the multi-round memory-harness env (Step 3: train the memory strategy).

RUN: .venv-tinker/bin/python scripts/tinker_grpo_mem.py --model ... --train ... \
       --log-path data/runs/gt_rlvr/tier4_mem_30b  (source main-repo .env first)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_memory_env import MemoryDatasetBuilder  # noqa: E402

from tinker_cookbook import checkpoint_utils  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402


async def amain(a):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=a.model, explicit_renderer_name=None, load_checkpoint_path=None)
    kl_ref = train.KLReferenceConfig(base_model=a.model) if a.kl > 0 else None
    config = train.Config(
        learning_rate=a.lr,
        model_name=a.model,
        max_tokens=a.max_tokens,            # per-ROUND generation budget
        kl_penalty_coef=a.kl,
        kl_reference_config=kl_ref,
        renderer_name=renderer_name,
        log_path=a.log_path,
        eval_every=0,
        save_every=a.save_every,
        dataset_builder=MemoryDatasetBuilder(
            train_path=a.train,
            batch_size=a.groups_per_batch,
            group_size=a.group_size,
            model_name_for_tokenizer=a.model,
            renderer_name=renderer_name,
            rounds=a.rounds,
        ),
    )
    await train.main(config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--train", default="data/runs/gt_rlvr/train_tier4_mem.jsonl")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group-size", type=int, default=8, dest="group_size")
    ap.add_argument("--groups-per-batch", type=int, default=24, dest="groups_per_batch")
    ap.add_argument("--max-tokens", type=int, default=1600, dest="max_tokens")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=10, dest="save_every")
    ap.add_argument("--log-path", default="data/runs/gt_rlvr/tier4_mem_30b", dest="log_path")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.groups_per_batch, a.group_size, a.rounds, a.max_tokens = 2, 4, 3, 600
    asyncio.run(amain(a))
