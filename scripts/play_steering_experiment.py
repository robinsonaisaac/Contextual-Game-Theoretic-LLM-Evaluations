#!/usr/bin/env python3
"""Run the activation-steering-in-games experiment.

For each (vector, alpha) condition and each seed, run a full match on a warm
GPU container with ALL seats steered uniformly by the vector at that alpha
(alpha=0 is the shared, unsteered baseline). Matches at the same seed share
the same environment across conditions, so differences are attributable to
the steering coefficient.

Cooperation vector : pd_full_v1     (Gemma 4 E4B-it, layer 16, mean_trace)
Trust vector       : pd_E4B_trust_v1 (Gemma 4 E4B-it, layer 16, mean_trace)

Logs are written to data/runs/game_steering_v1/<condition>/seed<seed>.jsonl
and a manifest.json records every match + its summary.

Usage:
    python3 scripts/play_steering_experiment.py --game one_night_werewolf \
        --n-players 5 --seeds 25 --alphas " -4,0,4" --out data/runs/game_steering_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

HF_ID = "google/gemma-4-E4B-it"
GPU_TIER_CLS = "SteeringWorker"          # E4B small tier
COOP_RUN_ID = "pd_full_v1"
TRUST_RUN_ID = "pd_E4B_trust_v1"
LAYER = 16
POSITION = "mean_trace"


def conditions(alphas, vectors=("coop", "trust"), skip_baseline=False):
    """Yield (label, run_id, layer, alpha). alpha=0 is a single shared baseline."""
    if not skip_baseline:
        yield ("baseline", None, None, 0.0)
    for a in alphas:
        if a == 0:
            continue
        if "coop" in vectors:
            yield (f"coop_a{a:+g}", COOP_RUN_ID, LAYER, float(a))
        if "trust" in vectors:
            yield (f"trust_a{a:+g}", TRUST_RUN_ID, LAYER, float(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="one_night_werewolf")
    ap.add_argument("--n-players", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--alphas", default="-4,0,4")
    ap.add_argument("--nego-rounds", type=int, default=2)
    ap.add_argument("--msgs-per-slot", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=600)
    ap.add_argument("--out", default="data/runs/game_steering_v1")
    ap.add_argument("--vectors", default="coop,trust",
                    help="comma list of vectors to sweep (coop,trust). trust was a "
                         "replicated null in ONW+SH; pass 'coop' for the lean design.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="omit the alpha=0 baseline condition (when one already exists)")
    ap.add_argument("--job-timeout", type=int, default=1800,
                    help="per-match .get() timeout in seconds (skip on timeout; never hang)")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    conds = list(conditions(alphas, tuple(args.vectors.split(",")),
                            skip_baseline=args.skip_baseline))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"nego_rounds": args.nego_rounds, "msgs_per_slot": args.msgs_per_slot}

    W = modal.Cls.from_name("safety", GPU_TIER_CLS)
    worker = W(model_name=HF_ID)
    treat = list(range(args.n_players))

    manifest = []

    def _entry_from_log(label, alpha, run_id, seed, log_path):
        """Reconstruct a manifest entry from an already-saved match log (resume).
        Returns None if the log has no terminal event (partial/corrupt)."""
        term = None
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("type") == "terminal":
                term = ev
        if term is None:
            return None
        rewards = term.get("rewards", [])
        winner = int(max(range(len(rewards)), key=lambda i: rewards[i])) if rewards else None
        return {"label": label, "alpha": alpha, "seed": seed, "run_id": run_id,
                "winner": winner, "rewards": rewards, "n_turns": term.get("turn"),
                "log": str(log_path)}

    # RESUME: reuse any completed match log already on disk; spawn only what's missing.
    to_spawn = []
    reused = 0
    for (label, run_id, layer, alpha) in conds:
        for seed in range(args.seeds):
            log_path = out / label / f"seed{seed}.jsonl"
            if log_path.exists() and log_path.stat().st_size > 0:
                e = _entry_from_log(label, alpha, run_id, seed, log_path)
                if e is not None:
                    manifest.append(e); reused += 1
                    continue
            fc = worker.play_steered_match.spawn(
                game_name=args.game, n_players=args.n_players,
                run_id=run_id, layer=layer, position=POSITION, alpha=alpha,
                treat_seats=treat, seed=seed, config_dict=cfg,
                max_new_tokens=args.max_new_tokens, temperature=0.7,
                max_turns=args.max_turns)
            to_spawn.append({"label": label, "run_id": run_id, "layer": layer,
                             "alpha": alpha, "seed": seed, "call_id": fc.object_id})

    print(f"[exp] resume: reused {reused} existing logs; spawned {len(to_spawn)} new "
          f"on game={args.game}", flush=True)
    (out / "jobs.json").write_text(json.dumps({"game": args.game,
        "n_players": args.n_players, "config": cfg, "jobs": to_spawn}, indent=2))

    def _flush_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _flush_manifest()  # persist reused entries immediately
    done, failed = 0, 0
    for j in to_spawn:
        fc = modal.FunctionCall.from_id(j["call_id"])
        try:
            res = fc.get(timeout=args.job_timeout)   # per-job timeout -> never hang
        except Exception as e:
            failed += 1
            print(f"[exp] ERR {j['label']} seed{j['seed']}: {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
            continue
        cdir = out / j["label"]; cdir.mkdir(exist_ok=True)
        log_path = cdir / f"seed{j['seed']}.jsonl"
        log_path.write_text(res["log"])
        manifest.append({**{k: j[k] for k in ("label", "alpha", "seed", "run_id")},
                         "winner": res["winner"], "rewards": res["rewards"],
                         "n_turns": res["n_turns"], "log": str(log_path)})
        done += 1
        _flush_manifest()   # incremental -> a later hang/crash never loses collected work
        if done % 10 == 0:
            print(f"[exp] collected {done}/{len(to_spawn)}", flush=True)

    print(f"[exp] done: {len(manifest)} matches in manifest "
          f"({reused} reused + {done} new, {failed} failed) -> {out}/manifest.json",
          flush=True)


if __name__ == "__main__":
    main()
