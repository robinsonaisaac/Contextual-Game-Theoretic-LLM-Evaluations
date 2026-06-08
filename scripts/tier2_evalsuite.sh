#!/bin/bash
# Tier-2 breadth eval. Usage: bash scripts/tier2_evalsuite.sh <base|rlvr> [--model-path tinker://...]
# Generation-heavy game ops eval at 8192 tokens (token-wall-safe). Transfer = held-out reasoning
# (>=3 measurable: boolean/dyck/mmlu) + bbh_hard; gsm8k = no-regression control.
set -e; TAG="$1"; shift; SPEC="$@"; TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/gt_rlvr
run(){ $PY scripts/tinker_eval.py --eval "$1" --corpus "$2" $SPEC --tokenizer $TOK --max-tokens "${3:-2048}" --out "$D/t2_${TAG}_$4.json" 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"; }
run freetext $D/eval_tier2_extrap.jsonl    8192 extrap
run freetext $D/eval_tier2_indomain.jsonl  8192 indomain
run freetext $D/eval_boolean_eval.jsonl    2048 boolean_eval
run dyck     $D/eval_dyck.jsonl            2048 dyck
run mmlu_pro $D/eval_mmlu_pro.jsonl        1536 mmlu_pro
run bbh_hard $D/eval_bbh_hard.jsonl        2048 bbh_hard
run gsm8k    data/runs/capability/gsm8k_eval.jsonl 1536 gsm8k
echo "[suite-2] $TAG done"
