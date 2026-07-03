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
from pathlib import Path

OUT = Path("data/runs/tier5")
TOK = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PY = ".venv-tinker/bin/python"
EXTRAP = {"trees_h127", "register_machine_h150", "graph_search_h130", "forward_chain_h130"}


def _run(corpus, spec, tag):
    out = OUT / f"gate_{tag}_{Path(corpus).stem}.json"
    subprocess.run([PY, "scripts/tinker_eval.py", "--eval", "ledger", "--corpus", corpus,
                    *spec, "--tokenizer", TOK, "--temperature", "0", "--max-tokens", "4096",
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
