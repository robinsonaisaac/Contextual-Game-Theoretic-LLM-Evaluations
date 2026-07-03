"""LoRA SFT on game-theory CoT data via the Tinker API.

RUN WITH THE TINKER VENV:  .venv-tinker/bin/python scripts/tinker_sft.py ...
(Tinker requires Python >=3.11; the repo's default 3.9 cannot import it.)

STATUS: written against the Tinker SDK 0.22 surface
(create_lora_training_client / forward_backward(cross_entropy) / optim_step /
save_weights_for_sampler). It is NOT yet smoke-tested because the TINKER_API_KEY
in .env currently returns 401. On the first authenticated run, verify two things
flagged inline below: (1) the base-model name (from get_server_capabilities),
and (2) the exact loss_fn_inputs key names for the cross_entropy loss.

Pipeline: load (prompt, completion) JSONL -> tokenize -> build Datums with the
loss masked to completion tokens only -> SFT epochs -> save a sampler checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tinker
from tinker import AdamParams, Datum, LoraConfig, ModelInput
from transformers import AutoTokenizer


def list_models():
    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    return [m.model_name for m in caps.supported_models]


def build_datum(tokenizer, prompt: str, completion: str, max_len: int) -> Datum:
    """Full-sequence tokens; loss weight 1 on completion tokens, 0 on the prompt."""
    p_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer(p_text, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (p_ids + c_ids)[:max_len]
    # next-token targets; weight only the completion region
    inp = ids[:-1]
    target = ids[1:]
    weights = [0.0] * (len(p_ids) - 1) + [1.0] * (len(c_ids))
    weights = weights[:len(target)]
    return Datum(
        model_input=ModelInput.from_ints(inp),
        # (1) VERIFY these key names against tinker docs on first run:
        loss_fn_inputs={"target_tokens": target, "weights": weights},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--train", default="data/runs/gametree/sft_train.jsonl")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")  # (2) confirm available
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id; defaults to base-model")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--save-name", default="gametree_sft_v1")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on first N examples")
    ap.add_argument("--ckpt-out", default="data/runs/gametree/sft_checkpoint.txt",
                    help="sidecar file to receive the tinker:// checkpoint path")
    args = ap.parse_args()

    if args.list_models:
        for m in list_models():
            print(m)
        return

    rows = [json.loads(l) for l in Path(args.train).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[sft] {len(rows)} train examples; base={args.base_model} rank={args.rank}")

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.base_model)

    data = [build_datum(tok, r["prompt"], r["completion"], args.max_len) for r in rows]
    for ep in range(args.epochs):
        for i in range(0, len(data), args.batch):
            batch = data[i:i + args.batch]
            tc.forward_backward(batch, "cross_entropy").result()
            tc.optim_step(AdamParams(learning_rate=args.lr)).result()
        print(f"[sft] epoch {ep+1}/{args.epochs} done")

    path = tc.save_weights_for_sampler(args.save_name).result().path
    print(f"[sft] saved sampler checkpoint: {path}")
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_out).write_text(path)


if __name__ == "__main__":
    main()
