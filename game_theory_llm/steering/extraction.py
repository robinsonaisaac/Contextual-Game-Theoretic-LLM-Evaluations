"""Activation extraction: generate a trace, then re-run with hooks to capture
residual-stream activations at three token positions for every layer.

Designed to be called from inside a Modal worker that has already loaded the
model. The functions here take `model`, `tokenizer`, and the list of layer
modules so they are agnostic to the exact module path.
"""

from __future__ import annotations

from typing import Any

import torch

from game_theory_llm.decision_parser import extract_decision

from .models import POSITION_KEYS


def generate_trace(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int | None = None,
) -> tuple[str, torch.Tensor]:
    """Sample a reasoning trace given a prompt.

    Returns (trace_text, full_token_ids), where full_token_ids is the
    concatenation of prompt and generated tokens (1D tensor on CPU).
    """
    if seed is not None:
        torch.manual_seed(seed)
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
    return trace_text, full_ids


def extract_activations(
    model,
    tokenizer,
    layers: torch.nn.ModuleList,
    prompt: str,
    trace_text: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Re-run prompt+trace through the model with hooks attached to every
    decoder block, capturing the residual stream at three positions.

    Returns activations[layer_idx][position_key] -> 1D bf16 tensor on CPU.
    """
    device = model.device
    full_text = prompt + trace_text
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    seq_len = full_ids.shape[1]
    if seq_len <= prompt_len:
        # Trace was empty or tokenizer collapsed it; capture only last_prompt.
        prompt_len = seq_len - 1
    last_prompt_idx = prompt_len - 1
    last_trace_idx = seq_len - 1
    trace_slice = slice(prompt_len, seq_len)

    # Storage for captured activations.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (batch=1, seq, hidden); detach + move to CPU + bf16.
            captured[layer_idx] = h[0].detach().to("cpu", torch.bfloat16)
            return output
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            model(full_ids)
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
