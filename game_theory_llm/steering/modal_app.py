"""Modal app for activation extraction and steering evaluation on Gemma 4 E4B-it.

Run a one-shot warmup discovery from your laptop:

    modal run game_theory_llm/steering/modal_app.py::warmup

This loads the model on an A100, attaches a no-op forward hook, and returns
the actual layer count, hidden dim, module path, and dtype. Use the returned
numbers to populate downstream extraction code.
"""

# Discovered by warmup() on 2026-05-05:
#   n_layers     = 42
#   hidden_dim   = 2560
#   vocab_size   = 262144
#   module_path  = "model.model.language_model.layers"
#   dtype        = bfloat16
#   model_class  = Gemma4ForConditionalGeneration (multimodal; text decoder
#                  nested under model.model.language_model)

from __future__ import annotations

import modal

MODEL_NAME = "google/gemma-4-E4B-it"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Steering deps
        "torch==2.5.1",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface_hub",
        "pyarrow>=14",
        # game_theory_llm runtime deps (its __init__ transitively imports
        # the analysis subpackage which uses scipy/statsmodels/etc.)
        "openai",
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.7",
        "statsmodels>=0.13",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "networkx",
        "python-dotenv",
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
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[warmup] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[warmup] gpu={torch.cuda.get_device_name(0)}", flush=True)

    print("[warmup] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[warmup] tokenizer loaded", flush=True)

    print("[warmup] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"[warmup] model loaded; class={type(model).__name__} "
          f"config_class={type(model.config).__name__}", flush=True)

    # Discover the TEXT decoder layer list. For multimodal models like
    # Gemma4ForConditionalGeneration, `model.named_modules()` will surface
    # `vision_tower` and `audio_tower` ModuleLists too, which we must avoid.
    # Try known text-decoder paths first; fall back to a filtered walk.
    layers = None
    module_path = None
    candidate_paths = [
        ("model.language_model.model.layers",
         lambda m: getattr(getattr(getattr(m, "language_model", None), "model", None), "layers", None)),
        ("model.language_model.layers",
         lambda m: getattr(getattr(m, "language_model", None), "layers", None)),
        ("model.model.language_model.layers",
         lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None)),
        ("model.model.layers",
         lambda m: getattr(getattr(m, "model", None), "layers", None)),
    ]
    for path, getter in candidate_paths:
        cand = getter(model)
        if isinstance(cand, torch.nn.ModuleList) and len(cand) >= 8:
            layers = cand
            module_path = path
            break

    if layers is None:
        # Filtered walk: skip anything under vision/audio towers.
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                if "vision" in name or "audio" in name:
                    continue
                layers = mod
                module_path = name
                break
        if layers is None:
            raise RuntimeError("Could not locate text decoder layer list")

    n_layers = len(layers)
    print(f"[warmup] discovered n_layers={n_layers} module_path={module_path}", flush=True)

    # Confirm hook attachment works on a real forward, and infer hidden_dim
    # from the captured tensor shape (robust to whatever the config nests).
    captured = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(h.shape)
        captured["dtype"] = str(h.dtype)
        return output  # no-op
    handle = layers[n_layers // 2].register_forward_hook(hook)
    try:
        print("[warmup] running test forward...", flush=True)
        ids = tokenizer("Hello, world.", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
        print("[warmup] forward complete", flush=True)
    finally:
        handle.remove()

    # hidden_dim is just the last dim of the captured residual stream.
    hidden_dim = captured["shape"][-1] if "shape" in captured else None

    # vocab_size: try top-level then nested text_config.
    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None and hasattr(model.config, "text_config"):
        vocab_size = getattr(model.config.text_config, "vocab_size", None)

    result = {
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "module_path": module_path,
        "dtype": "bfloat16",
        "config_class": type(model.config).__name__,
        "config_top_level_keys": sorted(k for k in vars(model.config) if not k.startswith("_"))[:30],
        "hook_ok": "shape" in captured,
        "hook_capture_shape": captured.get("shape"),
        "hook_capture_dtype": captured.get("dtype"),
    }

    import json
    print("===WARMUP_RESULT_JSON===")
    print(json.dumps(result, indent=2, default=str))
    print("===END_WARMUP_RESULT===")
    return result


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
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        # Discover the TEXT decoder layer list (skip vision/audio towers in
        # multimodal Gemma4ForConditionalGeneration).
        m = self.model
        candidate_paths = [
            getattr(getattr(getattr(m, "language_model", None), "model", None), "layers", None),
            getattr(getattr(m, "language_model", None), "layers", None),
            getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None),
            getattr(getattr(m, "model", None), "layers", None),
        ]
        self.layers = None
        for cand in candidate_paths:
            if isinstance(cand, torch.nn.ModuleList) and len(cand) >= 8:
                self.layers = cand
                break
        if self.layers is None:
            for name, mod in m.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
                    if "vision" in name or "audio" in name:
                        continue
                    self.layers = mod
                    break
            if self.layers is None:
                raise RuntimeError("Could not locate text decoder layer list")

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
        print(f"[extract] worker entered: {len(stories)} stories, run_dir={run_dir}", flush=True)
        print(f"[extract] n_layers={len(self.layers)}", flush=True)

        bundles, paths = [], []
        for i, s in enumerate(stories):
            print(f"[extract] story {i+1}/{len(stories)}: {s['story_id']} — generating trace...", flush=True)
            trace_text, full_ids, prompt_len = generate_trace(
                self.model, self.tokenizer, s["prompt"],
                max_new_tokens=s.get("max_new_tokens", 512),
                temperature=s.get("temperature", 0.7),
                seed=s.get("seed"),
            )
            decision = parse_decision(trace_text)
            cooperated = is_cooperative(decision, s["coop_choice"])
            print(f"[extract]   trace_len={len(full_ids)-prompt_len} decision={decision!r} "
                  f"cooperated={cooperated}", flush=True)
            print(f"[extract]   running hooked re-pass for activations...", flush=True)
            acts = extract_activations(
                self.model, self.layers, full_ids, prompt_len,
            )
            print(f"[extract]   captured {len(acts)} layers, saving bundle...", flush=True)
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
            print(f"[extract]   saved {path}", flush=True)

        print(f"[extract] writing index to {index_path}", flush=True)
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


@app.local_entrypoint()
def smoke(stories_path: str = "/tmp/smoke_story.jsonl",
          run_id: str = "smoke1",
          split: str = "train") -> None:
    """Run a tiny extract on a JSONL of stories (path is local).

    Usage:
        modal run game_theory_llm/steering/modal_app.py::smoke \
            --stories-path /tmp/smoke_story.jsonl --run-id smoke1 --split train

    Designed for use inside `modal run` so logs from the worker container
    stream back to the local terminal.
    """
    import json
    from pathlib import Path

    stories = [
        json.loads(line) for line in Path(stories_path).read_text().splitlines()
        if line.strip()
    ]
    print(f"[smoke] loaded {len(stories)} stories from {stories_path}", flush=True)
    print(f"[smoke] dispatching extract to Modal worker (run_id={run_id}, split={split})...",
          flush=True)
    worker = SteeringWorker()
    summary = worker.extract.remote(stories, run_id, split)
    print("===EXTRACT_RESULT_JSON===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("===END_EXTRACT_RESULT===", flush=True)
