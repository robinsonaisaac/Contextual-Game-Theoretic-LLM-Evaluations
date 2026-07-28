#!/usr/bin/env python3
"""Mixed-population steering in One Night Werewolf.

The published gameplay experiment steered every seat identically, which answers
"if you steer everyone, does collective behaviour shift?" but not the
deployment-relevant question: does a steered MINORITY move a population of
unsteered agents, or does the group absorb it?

Design. Alpha is fixed at the maximum-efficacy magnitude already established in
the uniform runs (|alpha| = 4), both signs, and the budget goes into population
structure instead. Because a Werewolf and a village-team player have opposed win
conditions, "steer this seat toward cooperation" means different things depending
on the card, so the treated set is described by TWO counts rather than one:

    w = number of Werewolves steered      (0, 1, 2)
    v = number of village-team seats steered (0, 1, 2, 3)

giving a 3x4 grid. (0,0) is the unsteered baseline; the other 11 combinations run
at alpha = -4 and +4, for 23 cells. Total steered k = w+v sweeps 1..5, so the
dose axis and the role-composition axis both fall out of one design.

Role structure is held fixed: ONW deals 8 cards to 5 seats + 3 centre, so the
number of Werewolves actually in play varies by seed (0, 1 or 2 -- the published
run contains a seed with no Werewolf at all). Roles are a deterministic function
of the seed, so we filter to seeds with exactly 2 seated Werewolves and select
treated seats offline.

Which specific seats get treated rotates with the seed so no cell is confounded
with a fixed seat position (seat order matters in ONW: vote order, speaking
order). Every match records its treated seats and their roles.

Usage:
    modal deploy game_theory_llm/steering/modal_app.py     # workers bake engine code
    python3 scripts/play_mixed_population_experiment.py \
        --seeds 50 --out data/runs/onw_mixed_v1
    python3 scripts/play_mixed_population_experiment.py --dry-run   # design + cost only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF_ID = "google/gemma-4-E4B-it"
GPU_TIER_CLS = "SteeringWorker"
COOP_RUN_ID = "pd_full_v1"
LAYER = 16
POSITION = "mean_trace"
N_PLAYERS = 5
REQUIRED_HIDDEN = 2          # hold the role structure fixed across all cells

# Per-game facts. `hidden` is the concealed-team role set (the analogue of ONW's
# Werewolves); `roles_attr` differs because the two engines name the field
# differently; `est_usd` and `timeout` come from scripts/modal_spend.py and the
# measured match durations of the uniform runs -- SH averages 48.8 min/match
# against ONW's 4.7, so the ONW-sized 30-minute job timeout would silently drop
# ~70% of SH matches while still paying for the GPU.
GAMES = {
    "one_night_werewolf": {
        "module": "game_theory_llm.play.games.one_night_werewolf",
        "cls": "OneNightWerewolf",
        "hidden": {"Werewolf"},
        "roles_attr": "initial_roles",
        "filter_seeds": True,    # ONW deals 8 cards to 5 seats: hidden count varies
        "est_usd": 0.24,
        "timeout": 1800,
    },
    "secret_hitler": {
        "module": "game_theory_llm.play.games.secret_hitler",
        "cls": "SecretHitler",
        "hidden": {"Fascist", "Hitler"},
        "roles_attr": "roles",
        "filter_seeds": False,   # always 3 Liberal / 1 Fascist / 1 Hitler
        "est_usd": 2.54,
        "timeout": 10800,
    },
}


def seat_roles(game_name: str, seed: int, n_players: int = N_PLAYERS):
    """Roles dealt to seats 0..n-1 for this seed. Deterministic: the engine draws
    from random.Random(seed) in runner.run_match, so this reproduces it exactly."""
    import importlib
    spec = GAMES[game_name]
    cls = getattr(importlib.import_module(spec["module"]), spec["cls"])
    state = cls(n_players=n_players).initial_state(random.Random(seed))
    return list(getattr(state, spec["roles_attr"]))


def build_seed_pool(game_name: str, n_seeds: int, n_players: int = N_PLAYERS,
                    scan_limit: int = 5000):
    """Seeds seating exactly REQUIRED_HIDDEN concealed-team players."""
    spec = GAMES[game_name]
    if not spec["filter_seeds"]:
        return list(range(n_seeds))
    pool = []
    for seed in range(scan_limit):
        roles = seat_roles(game_name, seed, n_players)
        if sum(1 for r in roles if r in spec["hidden"]) == REQUIRED_HIDDEN:
            pool.append(seed)
            if len(pool) == n_seeds:
                return pool
    raise SystemExit(f"only found {len(pool)} qualifying seeds in {scan_limit}")


def choose_treated(game_name: str, seed: int, w: int, v: int,
                   n_players: int = N_PLAYERS):
    """Pick w concealed-team seats and v majority-team seats for this seed.

    Rotates the choice by seed so a cell is never tied to one seat position
    (seat index drives vote and speaking order in both games)."""
    spec = GAMES[game_name]
    roles = seat_roles(game_name, seed, n_players)
    hidden = [i for i, r in enumerate(roles) if r in spec["hidden"]]
    majority = [i for i, r in enumerate(roles) if r not in spec["hidden"]]
    if w > len(hidden) or v > len(majority):
        raise ValueError(f"seed {seed}: cannot take w={w}, v={v} from {roles}")
    rot_h = [hidden[(i + seed) % len(hidden)] for i in range(len(hidden))]
    rot_m = [majority[(i + seed) % len(majority)] for i in range(len(majority))]
    treated = sorted(rot_h[:w] + rot_m[:v])
    return treated, roles


def design_cells():
    """(label, w, v, alpha). (0,0) is the shared unsteered baseline."""
    yield ("baseline", 0, 0, 0.0)
    for w in range(0, REQUIRED_HIDDEN + 1):
        for v in range(0, N_PLAYERS - REQUIRED_HIDDEN + 1):
            if w == 0 and v == 0:
                continue
            for alpha in (-4.0, 4.0):
                yield (f"w{w}v{v}_a{alpha:+g}", w, v, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="one_night_werewolf", choices=sorted(GAMES))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--nego-rounds", type=int, default=2)
    ap.add_argument("--msgs-per-slot", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=600)
    ap.add_argument("--out", default="data/runs/onw_mixed_v1")
    ap.add_argument("--job-timeout", type=int, default=None,
                    help="per-match .get() timeout; defaults to the game's measured need "
                         "(ONW 1800s, SH 10800s). SH matches average 48.8 min, so the "
                         "ONW value would silently drop most of the run.")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell labels to run (default: all 23). "
                         "Use --cells baseline for a cheap completion-rate pilot.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the design and cost estimate, spawn nothing")
    args = ap.parse_args()
    spec = GAMES[args.game]
    if args.job_timeout is None:
        args.job_timeout = spec["timeout"]

    cells = list(design_cells())
    if args.cells:
        want = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in cells if c[0] in want]
        missing = want - {c[0] for c in cells}
        if missing:
            raise SystemExit(f"unknown cell labels: {sorted(missing)}")
    pool = build_seed_pool(args.game, args.seeds)
    total = len(cells) * len(pool)

    print(f"[mixed] game={args.game}  {len(cells)} cells x {len(pool)} seeds = {total} matches")
    print(f"[mixed] seed pool ({REQUIRED_HIDDEN} concealed-team seats"
          f"{', filtered' if spec['filter_seeds'] else ', fixed by deal'}): "
          f"{pool[:8]}{' ...' if len(pool) > 8 else ''}")
    print(f"[mixed] est. GPU cost ~${total * spec['est_usd']:,.0f} "
          f"at ${spec['est_usd']}/match | job-timeout {args.job_timeout}s")
    print(f"[mixed] cells: {', '.join(c[0] for c in cells)}")

    if args.dry_run:
        print("\n[mixed] --- treated-seat assignment preview ---")
        for label, w, v, alpha in cells[:6]:
            for seed in pool[:3]:
                treated, roles = choose_treated(args.game, seed, w, v)
                shown = [f"{i}:{roles[i][:4]}" for i in treated]
                print(f"  {label:14s} seed{seed:<4d} treat={shown or '[]'}")
        print("\n[mixed] dry run - nothing spawned.")
        return

    import modal
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"nego_rounds": args.nego_rounds, "msgs_per_slot": args.msgs_per_slot}
    worker = modal.Cls.from_name("safety", GPU_TIER_CLS)(model_name=HF_ID)

    manifest, to_spawn, reused = [], [], 0
    for label, w, v, alpha in cells:
        for seed in pool:
            log_path = out / label / f"seed{seed}.jsonl"
            if log_path.exists() and log_path.stat().st_size > 0:
                # resume: only reuse logs that reached a terminal event
                term = None
                for line in log_path.read_text().splitlines():
                    if line.strip():
                        ev = json.loads(line)
                        if ev.get("type") == "terminal":
                            term = ev
                if term is not None:
                    treated, roles = choose_treated(args.game, seed, w, v)
                    manifest.append({"label": label, "w": w, "v": v, "alpha": alpha,
                                     "seed": seed, "treat_seats": treated,
                                     "roles": roles, "winner": None,
                                     "rewards": term.get("rewards", []),
                                     "n_turns": term.get("turn"),
                                     "log": str(log_path)})
                    reused += 1
                    continue
            treated, roles = choose_treated(args.game, seed, w, v)
            fc = worker.play_steered_match.spawn(
                game_name=args.game, n_players=N_PLAYERS,
                run_id=None if alpha == 0 else COOP_RUN_ID,
                layer=None if alpha == 0 else LAYER,
                position=POSITION, alpha=float(alpha),
                treat_seats=treated, seed=seed, config_dict=cfg,
                max_new_tokens=args.max_new_tokens, temperature=0.7,
                max_turns=args.max_turns)
            to_spawn.append({"label": label, "w": w, "v": v, "alpha": alpha,
                             "seed": seed, "treat_seats": treated, "roles": roles,
                             "call_id": fc.object_id})

    print(f"[mixed] reused {reused} existing logs; spawned {len(to_spawn)}", flush=True)
    (out / "jobs.json").write_text(json.dumps(
        {"game": args.game, "n_players": N_PLAYERS, "config": cfg,
         "seed_pool": pool, "jobs": to_spawn}, indent=2))

    def flush():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    flush()
    done = failed = 0
    for j in to_spawn:
        fc = modal.FunctionCall.from_id(j["call_id"])
        try:
            res = fc.get(timeout=args.job_timeout)
        except Exception as e:
            failed += 1
            print(f"[mixed] ERR {j['label']} seed{j['seed']}: "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        cdir = out / j["label"]
        cdir.mkdir(parents=True, exist_ok=True)
        log_path = cdir / f"seed{j['seed']}.jsonl"
        log_path.write_text(res["log"])
        manifest.append({k: j[k] for k in ("label", "w", "v", "alpha", "seed",
                                           "treat_seats", "roles")}
                        | {"winner": res["winner"], "rewards": res["rewards"],
                           "n_turns": res["n_turns"], "log": str(log_path),
                           "treat_seats_echo": res.get("treat_seats")})
        done += 1
        flush()
        if done % 25 == 0:
            print(f"[mixed] collected {done}/{len(to_spawn)}", flush=True)

    print(f"[mixed] done: {len(manifest)} matches ({reused} reused + {done} new, "
          f"{failed} failed) -> {out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
