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


@contextmanager
def multi_steering_hook(layers: torch.nn.ModuleList,
                        vecs: list[SteeringVector],
                        alpha: float):
    """Attach steering hooks on multiple layers simultaneously, all at the
    same `alpha`. Removes all hooks on exit. Use to evaluate combined
    steering directions (e.g. layer-25 and layer-27 directions added
    together).
    """
    handles = [
        layers[v.layer].register_forward_hook(
            make_steering_hook(v.direction, alpha, v.raw_norm)
        )
        for v in vecs
    ]
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()


def generate_with_hook(model, tokenizer, prompt: str, *,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       seed: int | None = None,
                       apply_chat_template: bool = True) -> str:
    """Generate a trace from prompt; the caller is responsible for any active hooks.

    When `apply_chat_template` is True (default), `prompt` is wrapped as a
    single user-turn chat message before tokenization.
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
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
