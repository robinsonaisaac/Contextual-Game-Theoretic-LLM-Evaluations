#!/bin/bash
# Tier-1b eval suite. Usage: bash scripts/tier1b_evalsuite.sh <base|rlvr> [--model-path tinker://...]
# Transfer test = held-out DEEPER gametree depths; plus the knowledge-free reasoning suite
# and the gsm8k no-regression control. Tag b=base / r=rlvr.
set -e; TAG="$1"; shift; SPEC="$@"; TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/gt_rlvr
run(){ $PY scripts/tinker_eval.py --eval "$1" --corpus "$2" $SPEC --tokenizer $TOK --max-tokens "${3:-2048}" --out "$D/t1b_${TAG}_$4.json" 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"; }
run freetext $D/eval_depth_extrap_deep.jsonl  3072 depth_extrap_deep
run freetext $D/eval_gametree_indomain.jsonl  3072 gametree_indomain
run freetext $D/eval_boolean_eval.jsonl       2048 boolean_eval
run freetext $D/eval_countdown.jsonl          2048 countdown
run dyck     $D/eval_dyck.jsonl               2048 dyck
run freetext $D/eval_ordering.jsonl           2048 ordering
run prontoqa $D/eval_prontoqa.jsonl           2048 prontoqa
run freetext $D/eval_knights_knaves.jsonl     2048 knights_knaves
run mmlu_pro $D/eval_mmlu_pro.jsonl           1536 mmlu_pro
run gsm8k    data/runs/capability/gsm8k_eval.jsonl 1536 gsm8k
echo "[suite-1b] $TAG done"
