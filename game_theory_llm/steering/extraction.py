"""Activation extraction: generate a trace, then re-run with hooks to capture
residual-stream activations at three token positions for every layer.

Designed to be called from inside a Modal worker that has already loaded the
model. The functions here take `model`, `tokenizer`, and the list of layer
modules so they are agnostic to the exact module path.
"""

from __future__ import annotations

import torch

from game_theory_llm.decision_parser import extract_decision


def generate_trace(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int | None = None,
    apply_chat_template: bool = True,
) -> tuple[str, torch.Tensor, int]:
    """Sample a reasoning trace given a prompt.

    Returns (trace_text, full_token_ids, prompt_len).
    `full_token_ids` is the 1D CPU tensor of prompt+generated tokens.
    `prompt_len` is the number of prompt tokens (use this to slice).

    When `apply_chat_template` is True (default), `prompt` is wrapped as a
    single user-turn chat message and tokenized through the model's chat
    template — required for Gemma-4-it style models.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if apply_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_ids = out[0].detach().cpu()
    trace_ids = full_ids[prompt_len:]
    trace_text = tokenizer.decode(trace_ids, skip_special_tokens=True)
    return trace_text, full_ids, prompt_len


def extract_activations(
    model,
    layers: torch.nn.ModuleList,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Re-run the full prompt+trace token sequence through the model with
    hooks attached to every decoder block, capturing the residual stream at
    three positions.

    `full_ids` is the 1D LongTensor returned by `generate_trace` (the actual
    concatenation of prompt and generated tokens — do NOT re-tokenize, since
    BPE merges across the boundary can shift token positions).
    `prompt_len` is the number of prompt tokens, also from `generate_trace`.

    Returns activations[layer_idx][position_key] -> 1D bf16 tensor on CPU.
    """
    device = model.device
    full_ids_2d = full_ids.unsqueeze(0).to(device)
    seq_len = full_ids_2d.shape[1]
    if seq_len <= prompt_len:
        # Trace was empty; capture only last_prompt position.
        prompt_len = seq_len - 1
    last_prompt_idx = prompt_len - 1
    last_trace_idx = seq_len - 1
    trace_slice = slice(prompt_len, seq_len)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h[0].detach().to("cpu", torch.bfloat16)
            return output
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            model(full_ids_2d)
    finally:
        for h in handles:
            h.remove()

    activations: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, h in captured.items():
        activations[layer_idx] = {
            "last_prompt": h[last_prompt_idx].clone(),
            "last_trace": h[last_trace_idx].clone(),
            "mean_trace": h[trace_slice].float().mean(0).to(torch.bfloat16),
        }
    return activations


def parse_decision(trace_text: str) -> str | None:
    """Wrap the existing extract_decision; returns 'A', 'B', or None."""
    return extract_decision(trace_text)


def is_cooperative(decision: str | None, coop_choice: str) -> bool:
    """Whether a decision matches the cooperative choice for this PD.

    coop_choice is the letter ('A' or 'B') the payoff matrix designates as
    the cooperative play. Returns False on None.
    """
    if decision is None:
        return False
    return decision == coop_choice
