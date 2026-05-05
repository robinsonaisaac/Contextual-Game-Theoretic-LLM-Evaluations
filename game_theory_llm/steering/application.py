"""Apply a steering vector at inference time via a forward hook on one layer.

Usage (inside a Modal worker that has model + tokenizer + layers):

    with steering_hook(layers, vec, alpha=1.0) as ctx:
        text = generate_with_hook(model, tokenizer, prompt)
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .models import SteeringVector


def make_steering_hook(direction: torch.Tensor, alpha: float, raw_norm: float):
    """Build a forward hook that adds (alpha * raw_norm) * direction to the
    layer output's residual stream at every token position."""
    scaled = (alpha * raw_norm) * direction
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + scaled.to(h.device, h.dtype)
            return (h, *output[1:])
        return output + scaled.to(output.device, output.dtype)
    return hook


@contextmanager
def steering_hook(layers: torch.nn.ModuleList, vec: SteeringVector, alpha: float):
    """Context manager: attach the steering hook on entry, remove on exit."""
    handle = layers[vec.layer].register_forward_hook(
        make_steering_hook(vec.direction, alpha, vec.raw_norm)
    )
    try:
        yield handle
    finally:
        handle.remove()


def generate_with_hook(model, tokenizer, prompt: str, *,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       seed: int | None = None) -> str:
    """Generate a trace from prompt; the caller is responsible for any active hooks."""
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
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
