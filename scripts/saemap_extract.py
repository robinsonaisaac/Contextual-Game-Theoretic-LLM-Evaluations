import argparse, json, modal
from game_theory_llm.saemap import corpus, paths
from game_theory_llm.steering import modal_app  # noqa: F401 (ensure app import)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-per-game", type=int, default=120)
    a = ap.parse_args()
    paths.ensure_run_dirs()
    dec = corpus.decision_pairs(limit=a.limit)
    dec_prompts = [d["prompt"] for d in dec]
    rec = corpus.recognition_sets(n_per_game=a.n_per_game)
    groups = {
        "decision_coop":    {"texts": dec_prompts,      "completion": corpus.COOP_CUE},
        "decision_defect":  {"texts": dec_prompts,      "completion": corpus.DEFECT_CUE},
        "recog_dilemma":    {"texts": rec["dilemma"],    "completion": None},
        "recog_nondilemma": {"texts": rec["nondilemma"], "completion": None},
        "recog_game":       {"texts": rec["game"],       "completion": None},
        "recog_nongame":    {"texts": rec["nongame"],    "completion": None},
    }
    w = modal.Cls.from_name("safety", "SaemapWorker")(model_name="saemap_9b")
    fc = w.extract_groups_to_volume.spawn(groups, paths.CANDIDATE_LAYERS, "saemap_9b")
    meta = {"call_id": fc.object_id, "groups": list(groups), "layers": paths.CANDIDATE_LAYERS,
            "decision_ids": [d["id"] for d in dec],
            "n": {k: len(v["texts"]) for k, v in groups.items()}}
    (paths.RUN_DIR / "extract_call.json").write_text(json.dumps(meta, indent=2))
    print("SPAWNED call_id =", fc.object_id)
    print("n per group:", meta["n"])
    print("You can disconnect now. Collect later with: .venv-sae/bin/python scripts/saemap_collect.py")

if __name__ == "__main__":
    main()
