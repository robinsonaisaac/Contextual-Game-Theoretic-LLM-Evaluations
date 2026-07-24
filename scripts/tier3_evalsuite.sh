#!/bin/bash
# Tier-3 eval suite. Usage: bash scripts/tier3_evalsuite.sh <base|outcome|process> [--model-path ...]
# PRIMARY: d6 extrapolation @8192. Budgets match tier2_evalsuite so t2_base_* transfer
# numbers are directly comparable for the base arm.
set -e; TAG="$1"; shift; SPEC="$@"; TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/gt_rlvr
run(){ $PY scripts/tinker_eval.py --eval "$1" --corpus "$2" $SPEC --tokenizer $TOK --max-tokens "${3:-2048}" --out "$D/t3_${TAG}_$4.json" 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"; }
run freetext $D/eval_t3_d6.jsonl        8192 d6
run freetext $D/eval_t3_indomain.jsonl  8192 indomain
run freetext $D/eval_boolean_eval.jsonl 2048 boolean_eval
run dyck     $D/eval_dyck.jsonl         2048 dyck
run mmlu_pro $D/eval_mmlu_pro.jsonl     1536 mmlu_pro
run bbh_hard $D/eval_bbh_hard.jsonl     2048 bbh_hard
run gsm8k    data/runs/capability/gsm8k_eval.jsonl 1536 gsm8k
echo "[suite-3] $TAG done"
