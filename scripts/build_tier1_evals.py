"""Build the Tier-1 knowledge-free, operation-isolated eval sets (op-tagged, depth-scaled)."""
import json
from pathlib import Path
from game_theory_llm.reasoning.eval_gen import dyck, prontoqa, countdown, ordering, knights_knaves, boolean_eval
OUT = Path("data/runs/gt_rlvr"); OUT.mkdir(parents=True, exist_ok=True)
GENS = {"dyck":dyck,"prontoqa":prontoqa,"countdown":countdown,"ordering":ordering,
        "knights_knaves":knights_knaves,"boolean_eval":boolean_eval}
def main():
    counts={}
    for name,fn in GENS.items():
        rows=[fn(seed=100*d+i, depth=d) for d in (2,3,4,5,6) for i in range(40)]
        for r in rows: r.setdefault("max_new_tokens", 256+256*int(r.get("depth",3)))
        (OUT/f"eval_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        counts[name]=len(rows)
    print(counts)
if __name__=="__main__": main()
