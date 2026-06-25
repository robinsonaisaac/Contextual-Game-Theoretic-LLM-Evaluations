"""Staged pipeline: decision2 contrast (natural decisions, no forced cue).

State dir: data/runs/saemap_9b/decision2/

Stages
------
label     -- elicit decisions from full_corpus; spawn generate job
build     -- grade continuations; balance coop/defect pool
extract   -- extract SAE residuals for balanced prompts via Modal
collect   -- pull activations from volume; SAE-encode per layer
discover  -- diff-of-means + L1 probe per layer; pick best
interpret -- max-activating examples for top features at best layer

Usage
-----
cd /Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering
source ../../.env
.venv-sae/bin/python scripts/saemap_decision2.py --stage label
.venv-sae/bin/python scripts/saemap_decision2.py --stage build
...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure package root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# State directory (relative to worktree)
# ---------------------------------------------------------------------------
from game_theory_llm.saemap import corpus, paths

STATE_DIR = paths.RUN_DIR / "decision2"
CORPUS_PATH = paths.WORKTREE / "data/runs/2026-05-05-sharp/steering/full_corpus.jsonl"
RUN_TAG = "saemap_9b_dec2"


def _ensure_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "raw").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage: label
# ---------------------------------------------------------------------------

def stage_label() -> None:
    """Load full_corpus.jsonl, build elicitation prompts, spawn generate job."""
    _ensure_state()

    # Load pool
    pool = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pool.append({
                "prompt": r["prompt"],
                "coop_letter": r["coop_choice"],
            })

    print(f"Pool size: {len(pool)}")

    # Build elicitation prompts (adds format header, hash-based example letter)
    elic = [corpus.decision_elicitation(x["prompt"]) for x in pool]

    # Spawn generate job on Modal worker
    import modal
    w = modal.Cls.from_name("safety", "SaemapWorker")(model_name="saemap_9b")
    fc = w.generate.spawn(elic, 2000, 0.0, 0, "</decision>")

    meta = {
        "call_id": fc.object_id,
        "pool": pool,
    }
    (STATE_DIR / "gen_call.json").write_text(json.dumps(meta, indent=2))

    print(f"GEN SPAWNED {fc.object_id}")
    print(f"Pool size: {len(pool)}")


# ---------------------------------------------------------------------------
# Stage: build
# ---------------------------------------------------------------------------

def stage_build() -> None:
    """Grade continuations with judge; balance coop/defect pool 50/50."""
    from game_theory_llm.saemap import judge

    meta = json.loads((STATE_DIR / "gen_call.json").read_text())
    call_id = meta["call_id"]
    pool = meta["pool"]

    import modal
    print(f"Waiting for generate job {call_id} ...")
    conts = modal.FunctionCall.from_id(call_id).get()
    print(f"Got {len(conts)} continuations")

    items = [
        {
            "scenario": x["prompt"],
            "continuation": c,
            "coop_letter": x["coop_letter"],
        }
        for x, c in zip(pool, conts)
    ]

    print(f"Grading {len(items)} items with LLM judge ...")
    verdicts = judge.grade_decisions(items)

    coop = []
    defect = []
    unclear_count = 0
    for x, v in zip(pool, verdicts):
        verdict = v["verdict"]
        if verdict == "cooperate":
            coop.append(x["prompt"])
        elif verdict == "defect":
            defect.append(x["prompt"])
        else:
            unclear_count += 1

    print(f"Raw counts — cooperate: {len(coop)}, defect: {len(defect)}, unclear: {unclear_count}")

    n = min(len(coop), len(defect))
    rng = random.Random(0)
    coop_bal = rng.sample(coop, n) if len(coop) > n else coop
    defect_bal = rng.sample(defect, n) if len(defect) > n else defect

    print(f"Balanced n_each={n}")

    balanced = {
        "coop": coop_bal,
        "defect": defect_bal,
        "n_each": n,
        "raw": {
            "cooperate": len(coop),
            "defect": len(defect),
            "unclear": unclear_count,
        },
    }
    (STATE_DIR / "balanced.json").write_text(json.dumps(balanced, indent=2))
    print("Wrote decision2/balanced.json")


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

def stage_extract() -> None:
    """Spawn Modal extract job for balanced coop/defect groups."""
    _ensure_state()

    balanced = json.loads((STATE_DIR / "balanced.json").read_text())
    coop = balanced["coop"]
    defect = balanced["defect"]

    groups = {
        "decision2_coop":   {"texts": coop,   "completion": None},
        "decision2_defect": {"texts": defect,  "completion": None},
    }

    import modal
    w = modal.Cls.from_name("safety", "SaemapWorker")(model_name="saemap_9b")
    fc = w.extract_groups_to_volume.spawn(groups, paths.CANDIDATE_LAYERS, RUN_TAG)

    meta = {
        "call_id": fc.object_id,
        "groups": list(groups.keys()),
        "layers": paths.CANDIDATE_LAYERS,
    }
    (STATE_DIR / "extract_call.json").write_text(json.dumps(meta, indent=2))

    print(f"EXTRACT SPAWNED {fc.object_id}")


# ---------------------------------------------------------------------------
# Stage: collect
# ---------------------------------------------------------------------------

def stage_collect() -> None:
    """Pull activations from Modal volume; SAE-encode per layer."""
    import subprocess
    import torch
    from game_theory_llm.saemap.sae import QwenScopeSAE

    meta = json.loads((STATE_DIR / "extract_call.json").read_text())
    call_id = meta["call_id"]
    groups = meta["groups"]
    layers = meta["layers"]

    import modal
    print(f"Waiting for extract job {call_id} ...")
    summary = modal.FunctionCall.from_id(call_id).get()
    print("Extract summary:", summary)

    raw_dir = STATE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Pull raw activation files from Modal volume
    for g in groups:
        remote_path = f"runs/{RUN_TAG}/activations/{g}.pt"
        local_path = raw_dir / f"{g}.pt"
        subprocess.check_call([
            ".venv-sae/bin/python", "-m", "modal", "volume", "get",
            "--force", "safety", remote_path, str(local_path),
        ])
        print(f"  Pulled {g}.pt")

    # SAE-encode each group at each layer
    for li, L in enumerate(layers):
        sae = QwenScopeSAE.load(L)
        out_dir = STATE_DIR / f"L{L}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for g in groups:
            data = torch.load(raw_dir / f"{g}.pt", map_location="cpu")
            resid = data["residuals"][:, li, :].float()   # [N, D_MODEL]
            encoded = sae.encode(resid)                    # [N, D_SAE]
            torch.save(encoded, out_dir / f"{g}.pt")

        print(f"L{L} encoded {len(groups)} groups")

    print("COLLECT DONE")


# ---------------------------------------------------------------------------
# Stage: discover
# ---------------------------------------------------------------------------

def stage_discover() -> None:
    """Diff-of-means + L1 probe per candidate layer; write summary."""
    import numpy as np
    import torch
    from game_theory_llm.saemap import discover

    layers = paths.CANDIDATE_LAYERS
    best = None

    for L in layers:
        pos = torch.load(STATE_DIR / f"L{L}" / "decision2_coop.pt",
                         map_location="cpu").numpy()       # [N, D_SAE]
        neg = torch.load(STATE_DIR / f"L{L}" / "decision2_defect.pt",
                         map_location="cpu").numpy()

        X = np.concatenate([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

        dom = discover.diff_of_means(pos, neg)
        probe = discover.l1_probe(X, y, C=0.05)

        coef_arr = np.array(probe["coef"])
        res = {
            "layer": L,
            "auc": probe["auc"],
            "n_nonzero": probe["n_nonzero"],
            "diffmeans_top": discover.top_features(dom, 10),
            "probe_top": [int(i) for i in np.argsort(-np.abs(coef_arr))[:10]],
        }

        out_path = STATE_DIR / f"discover_L{L}.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"L{L} auc={res['auc']:.3f} n_nz={res['n_nonzero']}")

        if best is None or res["auc"] > best["auc"]:
            best = res

    summary = {
        "best_layer": best["layer"],
        "best_auc": best["auc"],
        "n_nonzero": best["n_nonzero"],
        "diffmeans_top": best["diffmeans_top"],
    }
    (STATE_DIR / "discover_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best: L{best['layer']} auc={best['auc']:.3f}")
    print("DISCOVER DONE")


# ---------------------------------------------------------------------------
# Stage: interpret
# ---------------------------------------------------------------------------

def stage_interpret() -> None:
    """Max-activating examples for top-5 diffmeans features at best layer."""
    from game_theory_llm.saemap import interpret
    from game_theory_llm.saemap.sae import QwenScopeSAE

    summary = json.loads((STATE_DIR / "discover_summary.json").read_text())
    best_L = summary["best_layer"]
    top5 = summary["diffmeans_top"][:5]

    print(f"Best layer: {best_L}, top-5 features: {top5}")

    # Build interpretation pool
    game_texts = corpus.game_texts(
        paths.DILEMMA_GAMES + paths.NONDILEMMA_GAMES, 40
    )
    nongame_texts = corpus.nongame_texts(120)
    pool = game_texts + nongame_texts
    print(f"Interp pool: {len(game_texts)} game + {len(nongame_texts)} nongame = {len(pool)} total")

    sae = QwenScopeSAE.load(best_L)

    results = {}
    for f in top5:
        print(f"  Feature {f} ...")
        examples = interpret.max_activating_examples(
            layer=best_L,
            feature=f,
            texts=pool,
            sae=sae,
            topn=8,
        )
        results[str(f)] = examples
        if examples:
            print(f"    top score={examples[0]['score']:.4f} snippet={examples[0]['snippet'][:80]!r}")

    out_path = STATE_DIR / "interpret.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    print("INTERPRET DONE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "label": stage_label,
    "build": stage_build,
    "extract": stage_extract,
    "collect": stage_collect,
    "discover": stage_discover,
    "interpret": stage_interpret,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="decision2 staged pipeline — natural-decision contrast (no cue leakage)"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGES.keys()),
        help="Which stage to run",
    )
    args = parser.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
