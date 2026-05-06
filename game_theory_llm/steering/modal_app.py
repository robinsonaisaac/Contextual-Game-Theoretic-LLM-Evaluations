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
MODEL_LOCAL_PATH = "/data/models/gemma-4-E4B-it"

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


@app.function(volumes={"/data": volume}, timeout=120)
def list_volume(path: str = "runs/pd_full_v1_swap") -> dict:
    """Diagnostic: list contents of /data/<path> from a fresh container."""
    from pathlib import Path
    volume.reload()
    p = Path("/data") / path
    if not p.exists():
        out = {"path": str(p), "exists": False}
    else:
        items = sorted(str(x.relative_to("/data")) for x in p.rglob("*"))
        out = {"path": str(p), "exists": True, "items": items[:50]}
    print("===LIST_VOLUME_RESULT===")
    import json
    print(json.dumps(out, indent=2))
    print("===END_LIST_VOLUME_RESULT===")
    return out


@app.function(volumes={"/data": volume}, timeout=1800)
def download_model_once() -> dict:
    """Download the Gemma 4 E4B-it weights into the safety Volume so worker
    containers load the model from local disk instead of fetching from HF
    (which rate-limits 15 simultaneous unauthenticated downloads).

    Idempotent — skips if config.json already exists at MODEL_LOCAL_PATH.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(MODEL_LOCAL_PATH)
    if (target / "config.json").exists():
        n_files = sum(1 for _ in target.rglob("*"))
        return {"already_present": True, "path": str(target), "n_files": n_files}

    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] snapshot_download {MODEL_NAME} -> {target}", flush=True)
    snapshot_download(repo_id=MODEL_NAME, local_dir=str(target),
                      local_dir_use_symlinks=False)
    volume.commit()
    n_files = sum(1 for _ in target.rglob("*"))
    print(f"[download] done; {n_files} files", flush=True)
    return {"downloaded": True, "path": str(target), "n_files": n_files}


@app.function(volumes={"/data": volume}, timeout=600)
def rebuild_index(run_id: str, split: str = "train") -> dict:
    """Walk /data/runs/{run_id}/activations/{split}/*.pt and rewrite the
    Parquet index. Recovers from extract jobs that timed out before the
    index was written.
    """
    from pathlib import Path
    from game_theory_llm.steering.storage import (
        load_activation_bundle, write_index,
    )

    run_dir = Path(f"/data/runs/{run_id}")
    bundle_dir = run_dir / "activations" / split
    pts = sorted(bundle_dir.glob("*.pt"))
    print(f"[rebuild_index] found {len(pts)} bundles in {bundle_dir}", flush=True)
    bundles, paths = [], []
    for p in pts:
        b = load_activation_bundle(p)
        bundles.append(b)
        paths.append(p)
    index_path = run_dir / "index.parquet"
    write_index(bundles, paths, index_path, split=split)
    n_coop = sum(1 for b in bundles if b.cooperated)
    return {
        "n_bundles": len(bundles),
        "n_coop": n_coop,
        "n_defect": len(bundles) - n_coop,
        "index_path": str(index_path),
    }


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
    timeout=25200,
    scaledown_window=300,
)
class SteeringWorker:
    """Long-lived Modal container that loads Gemma 4 E4B-it once."""

    @modal.enter()
    def load(self):
        from pathlib import Path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Prefer the local volume copy (fast, no HF rate-limits).
        if Path(MODEL_LOCAL_PATH, "config.json").exists():
            src = MODEL_LOCAL_PATH
            print(f"[worker] loading model from volume: {src}", flush=True)
        else:
            src = MODEL_NAME
            print(f"[worker] volume copy missing; falling back to HF: {src}",
                  flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForCausalLM.from_pretrained(
            src,
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
    def get_unembedding(self) -> dict:
        """Return the lm_head weight matrix for logit attribution.

        Returns {"W_U": tensor (vocab × hidden_dim, float32), "vocab_size": int,
                 "hidden_dim": int}. The caller projects layer deltas onto the
                 A-vs-B direction: logit_diff_l = (W_U[A_id] - W_U[B_id]) @ delta_h_l.
        """
        import torch
        # lm_head may live directly on the model or inside language_model
        lm_head = (
            getattr(self.model, "lm_head", None)
            or getattr(getattr(self.model, "language_model", None), "lm_head", None)
            or getattr(getattr(getattr(self.model, "model", None), "language_model", None), "lm_head", None)
        )
        if lm_head is None:
            raise RuntimeError("Could not locate lm_head")
        W = lm_head.weight.detach().float().cpu()   # (vocab_size, hidden_dim)
        return {"W_U": W, "vocab_size": W.shape[0], "hidden_dim": W.shape[1]}

    @modal.method()
    def extract(self, stories: list[dict], run_id: str, split: str) -> dict:
        """Extract activations for a list of stories. Each story dict needs:
            story_id, prompt, coop_choice, (optional) seed, temperature.
        Optional mech-interp metadata fields (forwarded into bundle.metadata):
            game_type, contrast_dim, contrast_dim_level, cell_id.
        Bundles are written to /data/runs/{run_id}/activations/{split}/.
        """
        from pathlib import Path
        import pandas as pd
        from game_theory_llm.steering.extraction import (
            generate_trace, extract_activations, parse_decision, is_cooperative,
        )
        from game_theory_llm.steering.models import ActivationBundle
        from game_theory_llm.steering.storage import save_activation_bundle, write_index

        self._reload_volume()
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
                max_new_tokens=s.get("max_new_tokens", 768),
                temperature=s.get("temperature", 0.7),
                seed=s.get("seed"),
                apply_chat_template=s.get("apply_chat_template", True),
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
                    # Mech-interp fields (present when built by build_mech_interp_corpus.py)
                    "game_type": s.get("game_type"),
                    "contrast_dim": s.get("contrast_dim"),
                    "contrast_dim_level": s.get("contrast_dim_level"),
                    "cell_id": s.get("cell_id"),
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

    def _reload_volume(self):
        try:
            volume.reload()
        except Exception as e:
            print(f"[worker] volume.reload skipped: {e}", flush=True)

    @modal.method()
    def eval_shard_multi(self, run_id: str, cells: list[tuple[int, str]],
                         alpha: float, stories: list[dict],
                         label: str | None = None,
                         result_subdir: str = "shards_multi") -> dict:
        """Evaluate a single alpha applied to a *combination* of (layer, position)
        steering vectors simultaneously. The hook adds all directions at once.

        Each shard writes to /data/runs/{run_id}/shards_multi/{label}_a{alpha}.parquet
        (or auto-derives a label from the cells).
        """
        from pathlib import Path
        import json
        import time
        from game_theory_llm.steering.application import (
            multi_steering_hook, generate_with_hook,
        )
        from game_theory_llm.steering.extraction import parse_decision, is_cooperative
        from game_theory_llm.steering.models import SteeringEvalResult
        from game_theory_llm.steering.storage import load_vector_set, save_eval_results

        self._reload_volume()
        run_dir = Path(f"/data/runs/{run_id}")
        progress_path = run_dir / "eval_progress.jsonl"
        vs = load_vector_set(run_dir / "vectors.pt")
        cells_t = [tuple(c) for c in cells]
        vecs = [vs.vectors[c] for c in cells_t]
        if label is None:
            label = "+".join(f"L{l}_{p}" for l, p in cells_t)

        print(f"[shard_multi] cells={cells_t} alpha={alpha:+.1f} label={label} "
              f"n_stories={len(stories)}", flush=True)

        decisions = []
        n_coop = 0
        with multi_steering_hook(self.layers, vecs, alpha):
            for s in stories:
                t0 = time.time()
                text = generate_with_hook(
                    self.model, self.tokenizer, s["prompt"],
                    seed=s.get("seed"),
                    apply_chat_template=s.get("apply_chat_template", True),
                )
                d = parse_decision(text)
                cooperated = is_cooperative(d, s["coop_choice"])
                decisions.append({
                    "story_id": s["story_id"],
                    "decision": d,
                    "cooperated": cooperated,
                    "trace": text,
                })
                if cooperated:
                    n_coop += 1
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a") as f:
                    f.write(json.dumps({
                        "tag": "shard_multi",
                        "label": label,
                        "cells": [list(c) for c in cells_t],
                        "alpha": alpha,
                        "story_id": s["story_id"],
                        "decision": d,
                        "cooperated": cooperated,
                        "elapsed_s": time.time() - t0,
                        "trace": text,
                    }) + "\n")

        # Persist as a single-row "result" — layer/position columns hold the label
        # so aggregation is straightforward.
        result = SteeringEvalResult(
            layer=-1,
            position=label,
            alpha=alpha,
            n_stories=len(stories),
            n_cooperated=n_coop,
            cooperation_rate=n_coop / max(1, len(stories)),
            decisions=decisions,
        )
        out_path = run_dir / result_subdir / f"{label}_a{alpha:+.1f}.parquet"
        save_eval_results([result], out_path)
        print(f"[shard_multi] done; coop_rate={result.cooperation_rate:.3f} -> {out_path}",
              flush=True)
        return {
            "label": label,
            "cells": list(cells_t),
            "alpha": alpha,
            "n_stories": result.n_stories,
            "n_cooperated": result.n_cooperated,
            "cooperation_rate": result.cooperation_rate,
            "shard_path": str(out_path),
        }

    @modal.method()
    def eval_shard(self, run_id: str, layer: int, position: str, alpha: float,
                   stories: list[dict],
                   result_subdir: str = "shards") -> dict:
        """Evaluate one (layer, position, alpha) cell on `stories`.

        Each shard loads the SteeringVectorSet from the volume, picks the
        single (layer, position) vector, and runs steered generation at the
        given alpha across all stories. Writes per-shard results to
        /data/runs/{run_id}/shards/{layer}_{position}_{alpha}.parquet AND
        appends incremental progress to the shared eval_progress.jsonl.
        Designed to be fanned out via spawn() so an N-way split runs in
        parallel.
        """
        from pathlib import Path
        from game_theory_llm.steering.evaluation import _eval_one_cell
        from game_theory_llm.steering.storage import (
            load_vector_set, save_eval_results,
        )

        self._reload_volume()
        run_dir = Path(f"/data/runs/{run_id}")
        progress_path = run_dir / "eval_progress.jsonl"
        vs = load_vector_set(run_dir / "vectors.pt")
        vec = vs.vectors[(layer, position)]
        print(f"[shard] layer={layer} pos={position} alpha={alpha:+.1f} "
              f"n_stories={len(stories)}", flush=True)

        result = _eval_one_cell(
            self.model, self.tokenizer, self.layers, vec, alpha, stories,
            progress_path=progress_path, progress_tag="shard",
        )

        shard_path = (run_dir / result_subdir /
                      f"L{layer:02d}_{position}_a{alpha:+.1f}.parquet")
        save_eval_results([result], shard_path)
        print(f"[shard] done; coop_rate={result.cooperation_rate:.3f} "
              f"-> {shard_path}", flush=True)
        return {
            "layer": layer,
            "position": position,
            "alpha": alpha,
            "n_stories": result.n_stories,
            "n_cooperated": result.n_cooperated,
            "cooperation_rate": result.cooperation_rate,
            "shard_path": str(shard_path),
        }

    @modal.method()
    def evaluate(self, run_id: str, prune_stories: list[dict],
                 sweep_stories: list[dict] | None = None,
                 alpha_prune: float = 3.0,
                 keep_top_k: int = 5,
                 alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
                 layer_stride: int = 1,
                 positions: tuple[str, ...] | None = None,
                 survivor_override: tuple[tuple[int, str], ...] | None = None) -> dict:
        """Run prune-then-sweep evaluation against the SteeringVectorSet stored
        at /data/runs/{run_id}/vectors.pt.

        `prune_stories` is the (typically smaller) set used for the dense
        prune pass; `sweep_stories` is the (typically larger) set used for
        the alpha sweep on surviving cells. If `sweep_stories` is None, the
        prune set is reused for the sweep.

        `layer_stride` and `positions` filter the vector set BEFORE prune so
        we don't burn GPU on every (layer, position) cell on a first run.
        Stride 4 keeps every 4th layer; positions=("mean_trace",) keeps just
        that one residual-stream position.

        `survivor_override`, when set, skips the prune pass entirely and uses
        the given (layer, position) keys as the survivor set going into sweep.
        Useful when you've already eyeballed the strongest cells from
        raw_norms and want to save GPU.

        Per-cell-per-story progress is appended to
        /data/runs/{run_id}/eval_progress.jsonl so a disconnected client
        doesn't lose results.
        """
        from pathlib import Path
        from game_theory_llm.steering.evaluation import prune_pass, sweep_pass
        from game_theory_llm.steering.models import SteeringVectorSet
        from game_theory_llm.steering.storage import load_vector_set, save_eval_results

        run_dir = Path(f"/data/runs/{run_id}")
        progress_path = run_dir / "eval_progress.jsonl"
        vs_full = load_vector_set(run_dir / "vectors.pt")

        # Filter vectors before prune.
        kept = {}
        for (layer, pos), vec in vs_full.vectors.items():
            if layer_stride > 1 and (layer % layer_stride) != 0:
                continue
            if positions is not None and pos not in positions:
                continue
            kept[(layer, pos)] = vec
        vs = SteeringVectorSet(
            model_name=vs_full.model_name,
            vectors=kept,
            corpus_hash=vs_full.corpus_hash,
            fit_timestamp=vs_full.fit_timestamp,
        )
        print(f"[evaluate] vectors after filter: {len(kept)}/{len(vs_full.vectors)} "
              f"(layer_stride={layer_stride}, positions={positions})", flush=True)

        if sweep_stories is None:
            sweep_stories = prune_stories

        if survivor_override is not None:
            survivors = [tuple(k) for k in survivor_override]
            prune_results = []
            print(f"[evaluate] skipping prune; using {len(survivors)} override "
                  f"survivors: {survivors}", flush=True)
        else:
            survivors, prune_results = prune_pass(
                self.model, self.tokenizer, self.layers, vs, prune_stories,
                alpha_prune=alpha_prune, keep_top_k=keep_top_k,
                progress_path=progress_path,
            )
            print(f"[evaluate] prune complete; {len(survivors)} survivors", flush=True)

        sweep_results = sweep_pass(
            self.model, self.tokenizer, self.layers, vs, survivors, sweep_stories,
            alpha_grid=tuple(alpha_grid),
            progress_path=progress_path,
        )
        print(f"[evaluate] sweep complete", flush=True)

        prune_path = run_dir / "results_prune.parquet"
        sweep_path = run_dir / "results_sweep.parquet"
        if prune_results:
            save_eval_results(prune_results, prune_path)
        save_eval_results(sweep_results, sweep_path)

        return {
            "n_vectors_evaluated": len(kept),
            "n_survivors": len(survivors),
            "survivors": survivors,
            "prune_path": str(prune_path) if prune_results else None,
            "sweep_path": str(sweep_path),
        }


@app.local_entrypoint()
def eval_run(prune_path: str,
             sweep_path: str,
             run_id: str,
             alpha_prune: float = 3.0,
             keep_top_k: int = 5,
             alpha_grid: str = "-3,-2,-1,0,1,2,3",
             layer_stride: int = 1,
             positions: str = "mean_trace",
             survivor_override: str = "") -> None:
    """Run prune-then-sweep eval, streaming logs to local terminal.

    Usage:
        modal run game_theory_llm/steering/modal_app.py::eval_run \\
            --prune-path data/runs/.../prune.jsonl \\
            --sweep-path data/runs/.../eval.jsonl \\
            --run-id pd_full_v1 \\
            --layer-stride 4 --positions mean_trace
    """
    import json
    from pathlib import Path

    prune_stories = [json.loads(l) for l in Path(prune_path).read_text().splitlines() if l.strip()]
    sweep_stories = [json.loads(l) for l in Path(sweep_path).read_text().splitlines() if l.strip()]
    grid = tuple(float(x) for x in alpha_grid.split(","))
    pos_tuple = tuple(p.strip() for p in positions.split(",")) if positions else None
    override = None
    if survivor_override:
        override = tuple(
            (int(x.split(":")[0]), x.split(":")[1])
            for x in survivor_override.split(",")
        )

    print(f"[eval_run] {len(prune_stories)} prune stories, {len(sweep_stories)} sweep stories",
          flush=True)
    print(f"[eval_run] layer_stride={layer_stride} positions={pos_tuple} alpha_grid={grid}",
          flush=True)
    if override:
        print(f"[eval_run] survivor_override={override}", flush=True)

    worker = SteeringWorker()
    summary = worker.evaluate.remote(
        run_id=run_id,
        prune_stories=prune_stories,
        sweep_stories=sweep_stories,
        alpha_prune=alpha_prune,
        keep_top_k=keep_top_k,
        alpha_grid=grid,
        layer_stride=layer_stride,
        positions=pos_tuple,
        survivor_override=override,
    )
    print("===EVAL_RESULT_JSON===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("===END_EVAL_RESULT===", flush=True)


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
