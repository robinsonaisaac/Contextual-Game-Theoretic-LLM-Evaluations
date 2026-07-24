#!/bin/bash
set -e; TAG="$1"; shift; SPEC="$@"; TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/gt_rlvr
run(){ $PY scripts/tinker_eval.py --eval "$1" --corpus "$2" $SPEC --tokenizer $TOK --max-tokens "${3:-1024}" --out "$D/t1_${TAG}_$4.json" 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"; }
run freetext $D/eval_depth_extrap.jsonl  2048 depth_extrap
run freetext $D/eval_boolean_eval.jsonl  2048 boolean_eval
run freetext $D/eval_countdown.jsonl     2048 countdown
run dyck     $D/eval_dyck.jsonl          2048 dyck
run prontoqa $D/eval_prontoqa.jsonl      2048 prontoqa
run freetext $D/eval_ordering.jsonl      2048 ordering
run freetext $D/eval_knights_knaves.jsonl 2048 knights_knaves
run mmlu_pro $D/eval_mmlu_pro.jsonl      1536 mmlu_pro
run gsm8k    data/runs/capability/gsm8k_eval.jsonl 1536 gsm8k
echo "[suite] $TAG done"
