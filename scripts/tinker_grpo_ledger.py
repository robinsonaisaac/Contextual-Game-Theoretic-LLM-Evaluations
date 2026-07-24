"""GRPO on Tier-5 inline-ledger tasks via the Tinker cookbook rl.train loop.

RUN WITH:  .venv-tinker/bin/python scripts/tinker_grpo_ledger.py [--smoke] \
             --model-path <SFT tinker:// ckpt> --families <...> --horizons <...>
(source the main-repo .env first). Cram-boundary --families/--horizons come from
data/runs/tier5/gate_report.json (horizons where the SFT groups are mixed).

Pool selection (THIS governs):
  graph_search h60,90,130 | register_machine h60,90 | trees h31,63
  EXCLUDE: forward_chain (saturated at 1.0 -> zero advantage)
  EXCLUDE: register_machine h150 (never parses)

Full-run command (run detached by controller):
  SFT=$(cat data/runs/tier5/sft_checkpoint.txt)
  .venv-tinker/bin/python scripts/tinker_grpo_ledger.py \\
    --model-path "$SFT" \\
    --families graph_search,register_machine,trees \\
    --horizons 60,90,130,31,63 \\
    --max-tokens 6000 \\
    --save-every 10 \\
    --log-path data/runs/tier5/grpo_ledger_run
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_ledger_env import LedgerDatasetBuilder  # noqa: E402

from tinker_cookbook import checkpoint_utils  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402


async def amain(a):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=a.model, explicit_renderer_name=None, load_checkpoint_path=a.model_path or None)
    kl_ref = train.KLReferenceConfig(base_model=a.model) if a.kl > 0 else None
    config = train.Config(
        learning_rate=a.lr, model_name=a.model, max_tokens=a.max_tokens,
        kl_penalty_coef=a.kl, kl_reference_config=kl_ref, renderer_name=renderer_name,
        load_checkpoint_path=a.model_path or None,
        log_path=a.log_path, eval_every=a.eval_every, save_every=a.save_every,
        dataset_builder=LedgerDatasetBuilder(
            train_path=a.train, batch_size=a.groups_per_batch, group_size=a.group_size,
            model_name_for_tokenizer=a.model, renderer_name=renderer_name,
            families=a.families, horizons=a.horizons))
    await train.main(config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--model-path", default="", dest="model_path",
                    help="SFT tinker:// checkpoint to warm-start (RL continues from tier5_sft)")
    ap.add_argument("--train", default="data/runs/tier5/rl_pool.jsonl")
    ap.add_argument("--families", default="graph_search,register_machine,trees",
                    help="Comma-separated family filter (cram-boundary selection)")
    ap.add_argument("--horizons", default="60,90,130,31,63",
                    help="Comma-separated horizon filter (excludes register h150 and trees h127)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group-size", type=int, default=8, dest="group_size")
    ap.add_argument("--groups-per-batch", type=int, default=64, dest="groups_per_batch")
    ap.add_argument("--max-tokens", type=int, default=6000, dest="max_tokens",
                    help="h90 gold ≈3.4k tokens; 6000 leaves headroom")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=10, dest="save_every")
    ap.add_argument("--log-path", default="data/runs/tier5/grpo_ledger_run", dest="log_path")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny smoke run: 4 groups/batch, group_size=4, save_every=2, max_tokens=1024")
    a = ap.parse_args()
    if a.smoke:
        a.groups_per_batch, a.group_size = 4, 4
        a.eval_every, a.save_every, a.max_tokens = 0, 2, 1024
    asyncio.run(amain(a))
