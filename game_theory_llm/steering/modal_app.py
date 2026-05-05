"""Modal app for activation extraction and steering evaluation on Gemma 4 E4B-it.

Run a one-shot warmup discovery from your laptop:

    modal run game_theory_llm/steering/modal_app.py::warmup

This loads the model on an A100, attaches a no-op forward hook, and returns
the actual layer count, hidden dim, module path, and dtype. Use the returned
numbers to populate downstream extraction code.
"""

from __future__ import annotations

import modal

MODEL_NAME = "google/gemma-4-E4B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface_hub",
        "pandas>=2.0",
        "pyarrow>=14",
        "numpy>=1.24",
    )
    .add_local_python_source("game_theory_llm")
)

volume = modal.Volume.from_name("gtllm-steering", create_if_missing=True)

app = modal.App("gtllm-steering", image=image)


@app.function(gpu="A100", timeout=600)
def warmup() -> dict:
    """Discover model architecture by loading once and running a tiny forward.

    Returns a dict with: n_layers, hidden_dim, module_path, dtype, vocab_size.
    Also confirms forward_hook attachment works (returns hook_ok=True).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Discover module path. Try the two most likely candidates.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        module_path = "model.model.layers"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
        module_path = "model.language_model.layers"
    else:
        # Walk the module tree to find a list of identical decoder blocks.
        layers = None
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                layers = mod
                module_path = name
                break
        if layers is None:
            raise RuntimeError("Could not locate decoder layer list")

    n_layers = len(layers)
    hidden_dim = model.config.hidden_size

    # Confirm hook attachment works on a real forward.
    captured = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(h.shape)
        captured["dtype"] = str(h.dtype)
        return output  # no-op
    handle = layers[n_layers // 2].register_forward_hook(hook)
    try:
        ids = tokenizer("Hello, world.", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    return {
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "vocab_size": model.config.vocab_size,
        "module_path": module_path,
        "dtype": "bfloat16",
        "hook_ok": "shape" in captured,
        "hook_capture_shape": captured.get("shape"),
        "hook_capture_dtype": captured.get("dtype"),
    }
