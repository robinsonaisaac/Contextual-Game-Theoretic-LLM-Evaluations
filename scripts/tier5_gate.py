# scripts/tier5_gate.py
"""Tier-5 gate (spec §5): run base vs tier5_sft on the in-domain eval sets (trained +
EXTRAPOLATED horizons), aggregate accuracy, write gate_report.json, print the decision.

RUN WITH .venv-tinker after sourcing the main-repo .env:
  set -a; source .../.env; set +a
  .venv-tinker/bin/python scripts/tier5_gate.py"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier5_config import MAX_TOKEN_CEILING  # noqa: E402

OUT = Path("data/runs/tier5")
TOK = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PY = ".venv-tinker/bin/python"
EXTRAP = {"trees_h127", "register_machine_h150", "graph_search_h130", "forward_chain_h130"}

# Long sets need bigger budgets: gold canonical traces alone are ~3.4k-7.4k tokens, and
# the SFT model often writes in wordier naturalized styles. 4096 CENSORED these cells
# (truncation before the ANSWER tail -> parse 0), it did not measure them.
# MAX_TOKEN_CEILING is the single source of truth for this ceiling (see tier5_config.py);
# build_tier5_ledger.py's _MAXTOK imports the same constant so build/gate agree.
MAXTOK = {"trees_h127": MAX_TOKEN_CEILING, "register_machine_h150": MAX_TOKEN_CEILING,
          "register_machine_h90": MAX_TOKEN_CEILING, "graph_search_h130": MAX_TOKEN_CEILING,
          "forward_chain_h130": MAX_TOKEN_CEILING}


def _run(corpus, spec, tag):
    stem = Path(corpus).stem.replace("eval_indomain_", "")
    out = OUT / f"gate_{tag}_{Path(corpus).stem}.json"
    if out.exists():
        print(f"[gate] skipping {out.name} (already done)")
        return json.loads(out.read_text())["accuracy"]
    subprocess.run([PY, "scripts/tinker_eval.py", "--eval", "ledger", "--corpus", corpus,
                    *spec, "--tokenizer", TOK, "--temperature", "0",
                    "--max-tokens", str(MAXTOK.get(stem, 4096)),
                    "--out", str(out)], check=True)
    return json.loads(out.read_text())["accuracy"]


def main():
    sft_path = (OUT / "sft_checkpoint.txt").read_text().strip()
    sets = sorted(glob.glob(str(OUT / "eval_indomain_*.jsonl")))
    report = {"per_set": {}, "sft_path": sft_path}
    for corpus in sets:
        name = Path(corpus).stem.replace("eval_indomain_", "")
        base = _run(corpus, ["--base-model", TOK], "base")
        sft = _run(corpus, ["--model-path", sft_path], "sft")
        report["per_set"][name] = {"base_acc": base, "sft_acc": sft, "delta": sft - base}
    extrap = [v for n, v in report["per_set"].items() if n in EXTRAP]
    trained = [v for n, v in report["per_set"].items() if n not in EXTRAP]
    mean_extrap_delta = sum(v["delta"] for v in extrap) / max(1, len(extrap))
    mean_sft = sum(v["sft_acc"] for v in report["per_set"].values()) / max(1, len(report["per_set"]))
    mean_trained_sft = sum(v["sft_acc"] for v in trained) / max(1, len(trained))
    if mean_sft >= 0.90:
        decision = "SKIP-RL"       # evals saturated -> go to final eval
    elif mean_extrap_delta >= 0.15 or (mean_trained_sft >= 0.60 and mean_extrap_delta >= 0.05):
        decision = "PROCEED"       # behavior installed, reliability lags -> Phase 2
    else:
        decision = "STOP"          # ledger not installed even in-domain -> report the negative
    report["mean_extrap_delta"] = mean_extrap_delta
    report["mean_sft_acc"] = mean_sft
    report["mean_trained_sft_acc"] = mean_trained_sft
    report["decision"] = decision
    (OUT / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"GATE DECISION: {decision}")


if __name__ == "__main__":
    main()
