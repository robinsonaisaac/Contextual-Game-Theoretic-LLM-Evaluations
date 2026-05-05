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

volume = modal.Volume.from_name("safety", create_if_missing=True)

app = modal.App("safety", image=image)


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


@app.cls(
    gpu="A100",
    volumes={"/data": volume},
    timeout=3600,
    scaledown_window=300,
)
class SteeringWorker:
    """Long-lived Modal container that loads Gemma 4 E4B-it once."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        # Discover layers (mirror warmup logic).
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            self.layers = self.model.language_model.layers
        else:
            for _, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                    self.layers = mod
                    break
            else:
                raise RuntimeError("Could not locate decoder layer list")

    @modal.method()
    def extract(self, stories: list[dict], run_id: str, split: str) -> dict:
        """Extract activations for a list of stories. Each story dict needs:
            story_id, prompt, coop_choice, (optional) seed, temperature.
        Bundles are written to /data/runs/{run_id}/activations/{split}/.
        """
        from pathlib import Path
        import pandas as pd
        from game_theory_llm.steering.extraction import (
            generate_trace, extract_activations, parse_decision, is_cooperative,
        )
        from game_theory_llm.steering.models import ActivationBundle
        from game_theory_llm.steering.storage import save_activation_bundle, write_index

        run_dir = Path(f"/data/runs/{run_id}")
        bundle_dir = run_dir / "activations" / split
        index_path = run_dir / "index.parquet"

        bundles, paths = [], []
        for s in stories:
            trace_text, full_ids, prompt_len = generate_trace(
                self.model, self.tokenizer, s["prompt"],
                max_new_tokens=s.get("max_new_tokens", 512),
                temperature=s.get("temperature", 0.7),
                seed=s.get("seed"),
            )
            decision = parse_decision(trace_text)
            cooperated = is_cooperative(decision, s["coop_choice"])
            acts = extract_activations(
                self.model, self.layers, full_ids, prompt_len,
            )
            bundle = ActivationBundle(
                story_id=s["story_id"],
                model_name=MODEL_NAME,
                decision=decision or "",
                cooperated=cooperated,
                prompt_text=s["prompt"],
                trace_text=trace_text,
                activations=acts,
                metadata={
                    "temperature": s.get("temperature", 0.7),
                    "seed": s.get("seed"),
                    "coop_choice": s["coop_choice"],
                },
            )
            path = save_activation_bundle(bundle, bundle_dir)
            bundles.append(bundle)
            paths.append(path)

        write_index(bundles, paths, index_path, split=split)
        # Return summary only (don't ship bundles back over the wire).
        return {
            "n_stories": len(bundles),
            "decision_counts": pd.Series([b.decision for b in bundles]).value_counts().to_dict(),
            "coop_rate": sum(b.cooperated for b in bundles) / max(1, len(bundles)),
            "index_path": str(index_path),
        }

    @modal.method()
    def evaluate(self, run_id: str, stories: list[dict],
                 alpha_prune: float = 3.0,
                 keep_top_k: int = 10,
                 alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)) -> dict:
        """Run prune-then-sweep evaluation against the SteeringVectorSet stored
        at /data/runs/{run_id}/vectors.pt. Eval stories carry the same fields
        as extract() inputs."""
        from pathlib import Path
        from game_theory_llm.steering.evaluation import prune_pass, sweep_pass
        from game_theory_llm.steering.storage import load_vector_set, save_eval_results

        run_dir = Path(f"/data/runs/{run_id}")
        vs = load_vector_set(run_dir / "vectors.pt")

        survivors, prune_results = prune_pass(
            self.model, self.tokenizer, self.layers, vs, stories,
            alpha_prune=alpha_prune, keep_top_k=keep_top_k,
        )
        sweep_results = sweep_pass(
            self.model, self.tokenizer, self.layers, vs, survivors, stories,
            alpha_grid=tuple(alpha_grid),
        )

        prune_path = run_dir / "results_prune.parquet"
        sweep_path = run_dir / "results_sweep.parquet"
        save_eval_results(prune_results, prune_path)
        save_eval_results(sweep_results, sweep_path)

        return {
            "n_survivors": len(survivors),
            "survivors": survivors,
            "prune_path": str(prune_path),
            "sweep_path": str(sweep_path),
        }
