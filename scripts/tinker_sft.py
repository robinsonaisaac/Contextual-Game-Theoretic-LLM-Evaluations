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

    def _tofloats(v):
        """Coerce a tensor-like (TensorData/ndarray/list/scalar) to a flat float list."""
        if hasattr(v, "tolist"):
            v = v.tolist()
        elif hasattr(v, "data"):
            v = v.data
        if isinstance(v, (int, float)):
            return [float(v)]
        out = []
        for x in v:
            out.extend(_tofloats(x))
        return out

    def _batch_loss(fb_result):
        """Mean per-token loss from forward_backward's elementwise_loss; never raises."""
        try:
            tot, cnt = 0.0, 0
            for o in fb_result.loss_fn_outputs:
                v = None
                try:
                    v = o["elementwise_loss"]
                except Exception:
                    v = getattr(o, "elementwise_loss", None)
                if v is None:
                    continue
                fs = _tofloats(v)
                tot += sum(fs)
                cnt += len(fs)
            return tot / cnt if cnt else None
        except Exception:
            return None

    metrics_path = Path(args.ckpt_out).parent / "sft_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    n_batches = (len(data) + args.batch - 1) // args.batch
    step = 0
    for ep in range(args.epochs):
        for bi, i in enumerate(range(0, len(data), args.batch)):
            batch = data[i:i + args.batch]
            fb = tc.forward_backward(batch, "cross_entropy").result()
            tc.optim_step(AdamParams(learning_rate=args.lr)).result()
            step += 1
            loss = _batch_loss(fb)
            if step == 1:
                try:
                    print(f"[sft] loss_fn_output keys: {list(fb.loss_fn_outputs[0].keys())}",
                          flush=True)
                except Exception:
                    print("[sft] loss_fn_outputs introspection failed (logging loss=None)",
                          flush=True)
            print(f"[sft] epoch {ep+1}/{args.epochs} batch {bi+1}/{n_batches} "
                  f"loss {loss if loss is None else round(loss, 4)}", flush=True)
            try:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({"step": step, "epoch": ep + 1,
                                         "batch": bi + 1, "loss": loss}) + "\n")
            except Exception:
                pass  # metrics logging must never kill training
        print(f"[sft] epoch {ep+1}/{args.epochs} done", flush=True)

    path = tc.save_weights_for_sampler(args.save_name).result().path
    print(f"[sft] saved sampler checkpoint: {path}")
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_out).write_text(path)
    # ALSO save resumable training state: sampler checkpoints cannot warm-start
    # RL (load_weights rejects sampler_weights/ paths); RL needs a save_state path.
    state_path = tc.save_state(args.save_name + "_state").result().path
    print(f"[sft] saved training state: {state_path}")
    Path(str(args.ckpt_out) + ".state").write_text(state_path)


if __name__ == "__main__":
    main()
