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

from typing import List, Optional, Tuple

import modal

MODEL_NAME = "google/gemma-4-E4B-it"  # legacy default, kept for back-compat
MODEL_LOCAL_PATH = "/data/models/gemma-4-E4B-it"

# Cap on prompt length for in-container game play. Long games (Secret Hitler can
# run 200+ turns) otherwise grow the per-action prompt until a single attention
# allocation OOMs an 80GB GPU. 3072 keeps the recent-history tail + current
# decision while bounding peak memory; ONW prompts are far shorter so unaffected.
MAX_INPUT_TOKENS = 3072


def model_local_path(model_name: str) -> str:
    """Derive the per-model local cache path on the volume."""
    return f"/data/models/{model_name.split('/')[-1]}"

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
    # expandable_segments lets the CUDA caching allocator grow/shrink segments
    # instead of fragmenting, which is what the long-match OOM error suggested.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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


@app.function(volumes={"/data": volume}, timeout=3600)
def download_model_once(model_name: str = MODEL_NAME) -> dict:
    """Download a model into the safety Volume so worker containers load
    weights from local disk instead of fetching from HF.

    Idempotent — skips if config.json already exists at the model's local path.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(model_local_path(model_name))
    if (target / "config.json").exists():
        n_files = sum(1 for _ in target.rglob("*"))
        return {"already_present": True, "model": model_name,
                "path": str(target), "n_files": n_files}

    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] snapshot_download {model_name} -> {target}", flush=True)
    snapshot_download(repo_id=model_name, local_dir=str(target),
                      local_dir_use_symlinks=False)
    volume.commit()
    n_files = sum(1 for _ in target.rglob("*"))
    print(f"[download] done; {n_files} files", flush=True)
    return {"downloaded": True, "model": model_name,
            "path": str(target), "n_files": n_files}


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


def _warmup_impl(model_name: str) -> dict:
    """Shared warmup body. Loads model on the GPU of the calling function,
    discovers architecture, and runs a no-op hooked forward.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path

    print(f"[warmup] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[warmup] gpu={torch.cuda.get_device_name(0)}", flush=True)

    src = model_local_path(model_name)
    if not Path(src, "config.json").exists():
        src = model_name
        print(f"[warmup] volume copy missing; loading from HF: {src}", flush=True)
    else:
        print(f"[warmup] loading from volume: {src}", flush=True)

    print("[warmup] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(src)
    print("[warmup] tokenizer loaded", flush=True)

    print("[warmup] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src,
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
        "model_name": model_name,
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


@app.function(gpu="A100", volumes={"/data": volume}, timeout=600)
def warmup(model_name: str = MODEL_NAME) -> dict:
    """Warm up a model on A100-40GB. Use for ≤7B parameter models."""
    return _warmup_impl(model_name)


@app.function(gpu="A100-80GB", volumes={"/data": volume}, timeout=900)
def warmup_large(model_name: str) -> dict:
    """Warm up a large model on A100-80GB. Use for 26B/31B models."""
    return _warmup_impl(model_name)


def _discover_layers(model):
    """Find the text decoder ModuleList, skipping vision/audio towers."""
    import torch
    candidate_paths = [
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    ]
    for cand in candidate_paths:
        if isinstance(cand, torch.nn.ModuleList) and len(cand) >= 8:
            return cand
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            if "vision" in name or "audio" in name:
                continue
            return mod
    raise RuntimeError("Could not locate text decoder layer list")


def _worker_load_impl(self, model_name: str):
    """Shared SteeringWorker.load() body, parameterized by model_name."""
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = model_local_path(model_name)
    if Path(src, "config.json").exists():
        print(f"[worker] loading model from volume: {src}", flush=True)
    else:
        src = model_name
        print(f"[worker] volume copy missing; falling back to HF: {src}", flush=True)

    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(src)
    self.model = AutoModelForCausalLM.from_pretrained(
        src,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    self.model.eval()
    self.layers = _discover_layers(self.model)
    print(f"[worker] loaded {model_name}; n_layers={len(self.layers)}", flush=True)


def _safe_volume_reload():
    try:
        volume.reload()
    except Exception as e:
        print(f"[worker] volume.reload skipped: {e}", flush=True)


def _impl_extract(self, stories: list[dict], run_id: str, split: str) -> dict:
    from pathlib import Path
    import pandas as pd
    from game_theory_llm.steering.extraction import (
        generate_trace, extract_activations, parse_decision, is_cooperative,
    )
    from game_theory_llm.steering.models import ActivationBundle
    from game_theory_llm.steering.storage import save_activation_bundle, write_index

    _safe_volume_reload()
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
        # For reasoning corpora we contrast CORRECT vs INCORRECT solutions
        # (the "cooperated" flag is repurposed as the positive class for the
        # mean-difference fit), rather than cooperate vs defect.
        if s.get("game_type") == "gsm8k":
            from game_theory_llm.capability_scoring import gsm8k_correct
            cooperated = gsm8k_correct(trace_text, s.get("gsm8k_gold"))
        else:
            cooperated = is_cooperative(decision, s["coop_choice"])
        print(f"[extract]   trace_len={len(full_ids)-prompt_len} decision={decision!r} "
              f"cooperated={cooperated}", flush=True)
        acts = extract_activations(self.model, self.layers, full_ids, prompt_len)
        bundle = ActivationBundle(
            story_id=s["story_id"],
            model_name=self.model_name,
            decision=decision or "",
            cooperated=cooperated,
            prompt_text=s["prompt"],
            trace_text=trace_text,
            activations=acts,
            metadata={
                "temperature": s.get("temperature", 0.7),
                "seed": s.get("seed"),
                "coop_choice": s["coop_choice"],
                "game_type": s.get("game_type"),
                "contrast_dim": s.get("contrast_dim"),
                "contrast_dim_level": s.get("contrast_dim_level"),
                "cell_id": s.get("cell_id"),
            },
        )
        path = save_activation_bundle(bundle, bundle_dir)
        bundles.append(bundle)
        paths.append(path)

    write_index(bundles, paths, index_path, split=split)
    volume.commit()      # ensure bundles persist for sharded rebuild_index + fit
    return {
        "n_stories": len(bundles),
        "decision_counts": pd.Series([b.decision for b in bundles]).value_counts().to_dict(),
        "coop_rate": sum(b.cooperated for b in bundles) / max(1, len(bundles)),
        "index_path": str(index_path),
    }


def _impl_extract_prompt_only(self, stories: list[dict], run_id: str, split: str,
                               batch_size: int = 16) -> dict:
    from pathlib import Path
    import pandas as pd
    from game_theory_llm.steering.extraction import extract_prompt_activations_batched
    from game_theory_llm.steering.models import ActivationBundle
    from game_theory_llm.steering.storage import save_activation_bundle, write_index

    _safe_volume_reload()
    run_dir = Path(f"/data/runs/{run_id}")
    bundle_dir = run_dir / "activations" / split
    index_path = run_dir / "index.parquet"
    print(f"[extract_prompt_only] {len(stories)} stories, batch_size={batch_size}", flush=True)

    prompts = [s["prompt"] for s in stories]
    apply_ct = stories[0].get("apply_chat_template", True) if stories else True
    all_acts = extract_prompt_activations_batched(
        self.model, self.tokenizer, self.layers, prompts,
        batch_size=batch_size, apply_chat_template=apply_ct,
    )

    bundles, paths = [], []
    for s, acts in zip(stories, all_acts):
        bundle = ActivationBundle(
            story_id=s["story_id"],
            model_name=self.model_name,
            decision="",
            cooperated=False,
            prompt_text=s["prompt"],
            trace_text="",
            activations=acts,
            metadata={
                "coop_choice": s["coop_choice"],
                "game_type": s.get("game_type"),
                "contrast_dim": s.get("contrast_dim"),
                "contrast_dim_level": s.get("contrast_dim_level"),
                "cell_id": s.get("cell_id"),
            },
        )
        path = save_activation_bundle(bundle, bundle_dir)
        bundles.append(bundle)
        paths.append(path)
        print(f"[extract_prompt_only] saved {s['story_id']}", flush=True)

    write_index(bundles, paths, index_path, split=split)
    print(f"[extract_prompt_only] done. {len(bundles)} bundles written.", flush=True)
    return {"n_stories": len(bundles), "index_path": str(index_path)}


def _impl_eval_shard(self, run_id, layer, position, alpha, stories, result_subdir):
    from pathlib import Path
    from game_theory_llm.steering.evaluation import _eval_one_cell
    from game_theory_llm.steering.storage import load_vector_set, save_eval_results

    _safe_volume_reload()
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
    shard_path = run_dir / result_subdir / f"L{layer:02d}_{position}_a{alpha:+.1f}.parquet"
    save_eval_results([result], shard_path)
    print(f"[shard] done; coop_rate={result.cooperation_rate:.3f} -> {shard_path}", flush=True)
    return {
        "layer": layer, "position": position, "alpha": alpha,
        "n_stories": result.n_stories, "n_cooperated": result.n_cooperated,
        "cooperation_rate": result.cooperation_rate, "shard_path": str(shard_path),
    }


def _impl_eval_shard_multi(self, run_id, cells, alpha, stories, label, result_subdir):
    from pathlib import Path
    import json as _json
    import time
    from game_theory_llm.steering.application import multi_steering_hook, generate_with_hook
    from game_theory_llm.steering.extraction import parse_decision, is_cooperative
    from game_theory_llm.steering.models import SteeringEvalResult
    from game_theory_llm.steering.storage import load_vector_set, save_eval_results

    _safe_volume_reload()
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
                "story_id": s["story_id"], "decision": d,
                "cooperated": cooperated, "trace": text,
            })
            if cooperated:
                n_coop += 1
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a") as f:
                f.write(_json.dumps({
                    "tag": "shard_multi", "label": label,
                    "cells": [list(c) for c in cells_t],
                    "alpha": alpha, "story_id": s["story_id"],
                    "decision": d, "cooperated": cooperated,
                    "elapsed_s": time.time() - t0, "trace": text,
                }) + "\n")

    result = SteeringEvalResult(
        layer=-1, position=label, alpha=alpha,
        n_stories=len(stories), n_cooperated=n_coop,
        cooperation_rate=n_coop / max(1, len(stories)),
        decisions=decisions,
    )
    out_path = run_dir / result_subdir / f"{label}_a{alpha:+.1f}.parquet"
    save_eval_results([result], out_path)
    print(f"[shard_multi] done; coop_rate={result.cooperation_rate:.3f} -> {out_path}", flush=True)
    return {
        "label": label, "cells": list(cells_t), "alpha": alpha,
        "n_stories": result.n_stories, "n_cooperated": result.n_cooperated,
        "cooperation_rate": result.cooperation_rate, "shard_path": str(out_path),
    }


def _impl_play_steered_match(self, *, game_name, n_players, run_id, layer, position,
                             alpha, treat_seats, seed, config_dict,
                             max_new_tokens=384, temperature=0.7, max_turns=600):
    """Run ONE full game match entirely in-container.

    `treat_seats` is a list of seat indices that get the steering vector at
    coefficient `alpha`; all other seats generate with the same model at
    alpha=0 (a matched in-family baseline). The vector is
    /data/runs/{run_id}/vectors.pt at (layer, position). If run_id is None or
    alpha==0 everywhere, every seat is plain Gemma. Returns the full JSONL log
    text plus a result summary so the orchestrator never round-trips per action.
    """
    import json as _json
    from pathlib import Path
    from game_theory_llm.steering.application import steering_hook, generate_with_hook
    from game_theory_llm.steering.storage import load_vector_set
    from game_theory_llm.play import run_match, GameConfig
    from game_theory_llm.play.games import (
        OneNightWerewolf, SecretHitler, RiskLite, DiplomacyLite,
    )

    _safe_volume_reload()
    vec = None
    if run_id and layer is not None:
        vs = load_vector_set(Path(f"/data/runs/{run_id}/vectors.pt"))
        vec = vs.vectors[(layer, position)]

    cfg = GameConfig(**(config_dict or {}))
    if game_name == "one_night_werewolf":
        game = OneNightWerewolf(config=cfg, n_players=n_players)
    elif game_name == "secret_hitler":
        game = SecretHitler(cfg, n_players=n_players)
    elif game_name == "risk":
        game = RiskLite(cfg, n_players)
    elif game_name == "diplomacy":
        game = DiplomacyLite(cfg)            # standard 7 powers
    else:
        raise ValueError(f"unknown game {game_name}")

    model, tok, layers = self.model, self.tokenizer, self.layers
    treat = set(treat_seats or [])

    class _LocalSteeredPlayer:
        """In-process player: render -> generate (optionally hooked) -> text."""
        def __init__(self, seat, a):
            self.seat = seat
            self.alpha = float(a)
            self.name = f"Gemma[seat{seat},a={a:+.1f}]" if a else f"Gemma[seat{seat}]"
            self.history = []
            self._ncalls = 0

        def _gen(self, full, gseed):
            # min_new_tokens>0 stops a small model from returning an immediate
            # empty (EOS) completion on a forced-choice prompt. max_input_tokens
            # caps prompt length so long multi-turn games (Secret Hitler can run
            # 200+ turns) cannot grow the sequence until the GPU OOMs.
            if vec is not None and self.alpha != 0.0:
                with steering_hook(layers, vec, self.alpha):
                    return generate_with_hook(model, tok, full,
                                              max_new_tokens=max_new_tokens,
                                              temperature=temperature, seed=gseed,
                                              min_new_tokens=16,
                                              max_input_tokens=MAX_INPUT_TOKENS)
            return generate_with_hook(model, tok, full,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature, seed=gseed,
                                      min_new_tokens=16,
                                      max_input_tokens=MAX_INPUT_TOKENS)

        def act(self, g, state, idx):
            prompt = g.render_prompt(state, idx)
            full = self._build(prompt)
            self._ncalls += 1
            # Periodically release the caching allocator's reserved-but-unallocated
            # blocks; over a long match these fragment and can fail a large
            # contiguous attention allocation even when total free memory suffices.
            if self._ncalls % 16 == 0:
                import torch as _torch
                _torch.cuda.empty_cache()
            gseed = ((seed * 100003) ^ (idx * 7919) ^ (self._ncalls * 104729)) & 0x7FFFFFFF
            text = self._gen(full, gseed)
            # Empty / whitespace completion: retry once with a terse nudge and a
            # fresh seed so a transient degenerate sample can't force a fallback.
            if not text.strip():
                nudge = full + ("\n\nAnswer NOW with ONLY the exact tag the prompt "
                                "asks for and nothing else.")
                text = self._gen(nudge, (gseed * 2654435761) & 0x7FFFFFFF)
            self._push("you", prompt[-400:])
            self._push("reply", text)
            return text

        def receive_observation(self, obs):
            t = obs.get("type")
            if t == "message":
                frm = obs.get("from"); txt = obs.get("text", "")
                if obs.get("scope") == "private":
                    to = ",".join(f"P{x}" for x in obs.get("to", []))
                    self._push("chat", f"whisper P{frm}->[{to}]: {txt}")
                else:
                    self._push("chat", f"P{frm} (public): {txt}")
            elif t == "message_meta":
                self._push("chat", f"P{obs.get('from')} whispered to "
                                   f"{obs.get('n_recipients')} player(s)")
            elif t == "alliance_event":
                self._push("chat", f"alliance #{obs.get('alliance_id')} "
                                   f"{obs.get('event')} by P{obs.get('actor')}")
            elif t == "phase_change":
                self._push("chat", f"[phase -> {obs.get('to')}]")

        def _push(self, who, text):
            self.history.append((who, text))
            if len(self.history) > 80:
                self.history = self.history[-80:]

        def _build(self, current):
            parts = []
            if self.history:
                parts.append("=== recent game history ===")
                for who, text in self.history[-40:]:
                    parts.append(f"[{who}] {text}")
                parts.append("=== end history ===\n")
            parts.append(current)
            return "\n".join(parts)

    players = [_LocalSteeredPlayer(i, alpha if i in treat else 0.0)
               for i in range(game.n_players)]
    log = Path("/tmp/steered_match.jsonl")
    if log.exists():
        log.unlink()
    res = run_match(game, players, seed=seed, log_path=log, max_turns=max_turns)
    term = res.terminal_state
    return {
        "log": log.read_text(),
        "game": game_name,
        "n_players": game.n_players,
        "treat_seats": sorted(treat),
        "alpha": alpha,
        "run_id": run_id, "layer": layer, "position": position,
        "winner": getattr(term, "winner", getattr(term, "winner_team", None)),
        "rewards": res.rewards,
        "n_turns": res.n_turns,
    }


@app.cls(
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=43200,
    scaledown_window=300,
)
class SteeringWorker:
    """A100-80GB worker for Gemma 4 E2B/E4B extraction."""
    # Modal needs the type annotation as an actual type, not a string. With
    # `from __future__ import annotations` in effect we work around this by
    # importing str directly so Modal's typing.get_type_hints resolves it.
    model_name: str = modal.parameter(default=MODEL_NAME)

    @modal.enter()
    def load(self):
        _worker_load_impl(self, self.model_name)

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
        return _impl_extract(self, stories, run_id, split)

    @modal.method()
    def extract_prompt_only(self, stories: list[dict], run_id: str, split: str,
                            batch_size: int = 16) -> dict:
        return _impl_extract_prompt_only(self, stories, run_id, split, batch_size)

    def _reload_volume(self):
        _safe_volume_reload()

    @modal.method()
    def eval_shard_multi(self, run_id: str, cells: list[tuple[int, str]],
                         alpha: float, stories: list[dict],
                         label: Optional[str] = None,
                         result_subdir: str = "shards_multi") -> dict:
        return _impl_eval_shard_multi(self, run_id, cells, alpha, stories, label, result_subdir)

    @modal.method()
    def eval_shard(self, run_id: str, layer: int, position: str, alpha: float,
                   stories: list[dict],
                   result_subdir: str = "shards") -> dict:
        return _impl_eval_shard(self, run_id, layer, position, alpha, stories, result_subdir)

    @modal.method()
    def play_steered_match(self, game_name: str, n_players: int,
                           run_id: Optional[str], layer: Optional[int],
                           position: str, alpha: float,
                           treat_seats: list, seed: int,
                           config_dict: Optional[dict] = None,
                           max_new_tokens: int = 384, temperature: float = 0.7,
                           max_turns: int = 600) -> dict:
        return _impl_play_steered_match(
            self, game_name=game_name, n_players=n_players, run_id=run_id,
            layer=layer, position=position, alpha=alpha, treat_seats=treat_seats,
            seed=seed, config_dict=config_dict, max_new_tokens=max_new_tokens,
            temperature=temperature, max_turns=max_turns)

    @modal.method()
    def evaluate(self, run_id: str, prune_stories: list[dict],
                 sweep_stories: Optional[List[dict]] = None,
                 alpha_prune: float = 3.0,
                 keep_top_k: int = 5,
                 alpha_grid: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
                 layer_stride: int = 1,
                 positions: Optional[Tuple[str, ...]] = None,
                 survivor_override: Optional[Tuple[Tuple[int, str], ...]] = None) -> dict:
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


@app.cls(
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=43200,
    scaledown_window=300,
)
class SteeringWorkerLarge:
    """A100-80GB worker for >10B param models (Gemma 4 26B-A4B / 31B).

    Methods are thin wrappers that delegate to the SAME implementation
    bodies used by SteeringWorker, by inheriting from a plain Python mixin.
    To avoid Modal class-inheritance gotchas we just duplicate the method
    signatures here and call self._reload_volume / instance attributes.

    The actual heavy logic is identical to SteeringWorker — see those
    methods for documentation.
    """
    model_name: str = modal.parameter(default="google/gemma-4-26B-A4B-it")

    @modal.enter()
    def load(self):
        _worker_load_impl(self, self.model_name)

    def _reload_volume(self):
        try:
            volume.reload()
        except Exception as e:
            print(f"[worker-large] volume.reload skipped: {e}", flush=True)

    @modal.method()
    def extract(self, stories: list[dict], run_id: str, split: str) -> dict:
        return _impl_extract(self, stories, run_id, split)

    @modal.method()
    def eval_shard(self, run_id: str, layer: int, position: str, alpha: float,
                   stories: list[dict],
                   result_subdir: str = "shards") -> dict:
        return _impl_eval_shard(self, run_id, layer, position, alpha, stories, result_subdir)

    @modal.method()
    def play_steered_match(self, game_name: str, n_players: int,
                           run_id: Optional[str], layer: Optional[int],
                           position: str, alpha: float,
                           treat_seats: list, seed: int,
                           config_dict: Optional[dict] = None,
                           max_new_tokens: int = 384, temperature: float = 0.7,
                           max_turns: int = 600) -> dict:
        return _impl_play_steered_match(
            self, game_name=game_name, n_players=n_players, run_id=run_id,
            layer=layer, position=position, alpha=alpha, treat_seats=treat_seats,
            seed=seed, config_dict=config_dict, max_new_tokens=max_new_tokens,
            temperature=temperature, max_turns=max_turns)

    @modal.method()
    def eval_shard_multi(self, run_id: str, cells: list[tuple[int, str]],
                         alpha: float, stories: list[dict],
                         label: Optional[str] = None,
                         result_subdir: str = "shards_multi") -> dict:
        return _impl_eval_shard_multi(self, run_id, cells, alpha, stories, label, result_subdir)


# --- SAE feature-mapping (saemap) additions -------------------------------
# Qwen3.5-9B-Base does NOT load under transformers>=4.46 (the existing `image`
# pin, required by Gemma). It needs transformers ~5.x (config nests the decoder
# under text_config). Define a SEPARATE image so we never regress the Gemma
# workers. Same base + deps; only the transformers pin differs.
saemap_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=5.10",          # Qwen3.5 support (local validated 5.12.1)
        "accelerate>=1.0",
        "huggingface_hub",
        "safetensors",
        "numpy>=1.24",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("game_theory_llm")
)

SAEMAP_MODEL_NAME = "saemap_9b"          # volume dir = /data/models/saemap_9b
SAEMAP_HF_ID = "Qwen/Qwen3.5-9B-Base"


@app.function(image=saemap_image, volumes={"/data": volume}, timeout=3600)
def download_saemap_model(hf_id: str = SAEMAP_HF_ID,
                          local_name: str = SAEMAP_MODEL_NAME) -> dict:
    """Snapshot Qwen3.5-9B-Base into the safety volume at /data/models/<local_name>.

    Mirrors download_model_once but writes under a stable local_name so the worker
    loads from /data/models/saemap_9b regardless of the HF repo id. Idempotent.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(model_local_path(local_name))   # /data/models/saemap_9b
    if (target / "config.json").exists():
        n_files = sum(1 for _ in target.rglob("*"))
        return {"already_present": True, "hf_id": hf_id,
                "path": str(target), "n_files": n_files}
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download_saemap] {hf_id} -> {target}", flush=True)
    snapshot_download(repo_id=hf_id, local_dir=str(target),
                      local_dir_use_symlinks=False)
    volume.commit()
    n_files = sum(1 for _ in target.rglob("*"))
    print(f"[download_saemap] done; {n_files} files", flush=True)
    return {"downloaded": True, "hf_id": hf_id, "path": str(target), "n_files": n_files}


@app.cls(
    gpu="A100-80GB",
    image=saemap_image,
    volumes={"/data": volume},
    timeout=43200,
    scaledown_window=300,
)
class SaemapWorker:
    """A100-80GB MODEL-ONLY worker for Qwen3.5-9B-Base SAE feature-mapping.

    Loads ONLY the model (never the SAE — the controller holds the SAE locally).
    Captures residuals via forward hooks on self.layers[L] (same approach as
    steering/extraction.py), reads P(coop) from next-token logits over {A,B},
    and applies a residual-ADD steering hook for causal P(coop).
    """
    model_name: str = modal.parameter(default=SAEMAP_MODEL_NAME)

    @modal.enter()
    def load(self):
        _worker_load_impl(self, self.model_name)   # sets self.model/tokenizer/layers
        # Resolve a set of {A,B} token ids covering bare, space-prefixed, and
        # newline-prefixed forms. The model may put its mass on any of these
        # depending on context (e.g. " A" after a space, or "\nA" after newline).
        # add_special_tokens=False so we get the raw letter token.
        def _resolve_ids(letter):
            ids = set()
            for v in [letter, f" {letter}"]:          # bare and space-prefixed forms only
                toks = self.tokenizer(v, add_special_tokens=False).input_ids
                if toks:
                    ids.add(toks[-1])                  # the letter token in each form
            # keep ONLY ids that decode to the bare letter — excludes any boundary token
            ids = {t for t in ids if self.tokenizer.decode([t]).strip() == letter}
            return sorted(ids)

        self.tid_A = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        self.tid_B = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        self.tids_A = _resolve_ids("A")
        self.tids_B = _resolve_ids("B")
        print(f"[saemap] bare A/B ids = {self.tid_A}/{self.tid_B} | widened A/B = {self.tids_A}/{self.tids_B}", flush=True)

    # --- internal: one hooked forward, residuals at requested layers ---------
    def _residuals_one(self, text: str, layers: list, span: slice):
        """Return [len(layers), D_MODEL] float32: residual mean-pooled over `span`."""
        import torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        captured = {}

        def make_hook(L):
            def hook(module, inp):
                # inp is a tuple; inp[0] is the residual stream entering the block
                h = inp[0]
                captured[L] = h[0].detach()       # [T, D_MODEL] on device
            return hook

        handles = [self.layers[L].register_forward_pre_hook(make_hook(L)) for L in layers]
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for h in handles:
                h.remove()
        rows = []
        for L in layers:
            h = captured[L]                       # [T, D_MODEL]
            rows.append(h[span].float().mean(0).cpu())
        return torch.stack(rows)                  # [len(layers), D_MODEL]

    @modal.method()
    def extract_residuals(self, prompts: list[str], layers: list[int],
                          completion: str | None = None) -> list:
        """One forward per prompt; residual mean-pooled over the completion span
        (if `completion` given) else over all prompt tokens. Returns a nested list
        [N, len(layers), D_MODEL] (float32)."""
        import torch
        out = []
        for p in prompts:
            if completion is not None:
                pid = self.tokenizer(p, return_tensors="pt").input_ids
                start = pid.shape[1]
                text = p + completion
                full = self.tokenizer(text, return_tensors="pt").input_ids
                span = slice(start, full.shape[1])
            else:
                pid = self.tokenizer(p, return_tensors="pt").input_ids
                text = p
                span = slice(0, pid.shape[1])
            r = self._residuals_one(text, layers, span)   # [len(layers), D_MODEL]
            out.append(r.tolist())
        return out

    # --- internal: P(coop) for one text under whatever hooks are active ------
    def _pcoop_text(self, text: str, coop_letter: str) -> float:
        import torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        a = logits[self.tid_A].item()
        b = logits[self.tid_B].item()
        pa, pb = torch.softmax(torch.tensor([a, b]), 0).tolist()
        return pa if coop_letter == "A" else pb

    @modal.method()
    def pcoop(self, prompts: list[str], coop_letters: list[str],
              fewshot: str = "") -> list:
        """Normalized P(coop) over {A,B} after appending '\\n<decision>'."""
        return [self._pcoop_text(fewshot + p + "\n<decision>", c)
                for p, c in zip(prompts, coop_letters)]

    @modal.method()
    def ab_mass(self, prompts: list[str], fewshot: str = "") -> list:
        """Fraction of next-token mass on {A,B} after '\\n<decision>' (Step-0 gate).

        Sums over all A-variant and B-variant token ids (bare, space-prefixed,
        newline-prefixed) so we capture mass regardless of the tokenizer's
        boundary-sensitive encoding of the response letter.
        """
        import torch
        out = []
        for p in prompts:
            text = fewshot + p + "\n<decision>"
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                probs = self.model(ids).logits[0, -1].softmax(-1)
            mass = sum(probs[t].item() for t in self.tids_A) + \
                   sum(probs[t].item() for t in self.tids_B)
            out.append(mass)
        return out

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int = 1500,
                 temperature: float = 0.0, seed: int = 0,
                 stop_string: str | None = None) -> list:
        """Generate free-text continuations for each prompt (greedy by default).

        Used by the generate-and-judge decision readout in Step-0 gate: the base
        model generates a decision paragraph, which Sonnet then grades as
        cooperate/defect/unclear.

        If `stop_string` is set (e.g. "</decision>"), generation is truncated to
        include up to and including the first occurrence of that string. This
        keeps token budgets manageable for Qwen3.5-9B-Base which emits long
        <think> chains before its final <decision> tag.
        """
        import torch
        outs = []
        for p in prompts:
            ids = self.tokenizer(p, return_tensors="pt").input_ids.to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            if temperature and temperature > 0:
                torch.manual_seed(seed)
                kw.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(ids, **kw)
            text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            if stop_string and stop_string in text:
                text = text[:text.index(stop_string) + len(stop_string)]
            outs.append(text)
        return outs

    @modal.method()
    def causal_pcoop(self, prompts: list[str], coop_letters: list[str],
                     layer: int, vec: list[float], fewshot: str = "") -> list:
        """Same readout as pcoop but ADD the 4096-d `vec` to self.layers[layer]'s
        residual on every forward (mirrors make_steering_hook; inlined to avoid
        the pandas-dependent game_theory_llm import chain in saemap_image).

        Injection locus MUST match extract_residuals' capture locus (block input,
        i.e. resid_pre[L]), so we use register_forward_pre_hook on self.layers[layer].
        """
        import torch
        v = torch.tensor(vec, dtype=torch.float32)

        # Inject at block INPUT (resid_pre[L]) to match extract_residuals' capture locus.
        def pre_hook(module, args):
            h = args[0]
            return (h + v.to(h.device, h.dtype),) + tuple(args[1:])

        handle = self.layers[layer].register_forward_pre_hook(pre_hook)
        try:
            return [self._pcoop_text(fewshot + p + "\n<decision>", c)
                    for p, c in zip(prompts, coop_letters)]
        finally:
            handle.remove()


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
