"""Step-0 HARD GATE: is P(coop) a usable, responsive readout on 9B-Base?

Runs entirely against the Modal SaemapWorker via game_theory_llm.saemap.remote.
"""
import json, statistics
from game_theory_llm.saemap import remote, corpus, paths

FEWSHOT = (
    "You are deciding in a strategic scenario. Output your choice as a single letter.\n"
    "Example 1 ... <decision>A</decision>\n"
    "Example 2 ... <decision>B</decision>\n\n")

def run(fewshot):
    rows = corpus.pd_eval_set(limit=40)
    prompts = [r["prompt"] for r in rows]
    coop = [r["coop_letter"] for r in rows]
    ps = remote.pcoop(prompts, coop, fewshot).tolist()
    masses = remote.ab_mass(prompts, fewshot).tolist()
    return ps, masses

def main():
    paths.ensure_run_dirs()
    fewshot, used = "", False
    ps, masses = run(fewshot)
    if statistics.mean(masses) <= 0.5:
        fewshot, used = FEWSHOT, True
        ps, masses = run(fewshot)
    rec = {"n": len(ps), "mean_p_coop": statistics.mean(ps),
           "std_p_coop": statistics.pstdev(ps),
           "ab_mass_mean": statistics.mean(masses), "fewshot_used": used}
    rec["pass"] = rec["ab_mass_mean"] > 0.5 and rec["std_p_coop"] > 0.05
    (paths.RUN_DIR / "sanity.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print("GATE PASS" if rec["pass"] else "GATE FAIL")

if __name__ == "__main__":
    main()
