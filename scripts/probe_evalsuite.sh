#!/bin/bash
set -e; TAG="$1"; shift; SPEC="$@"; TOK="Qwen/Qwen3-4B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/gt_rlvr
$PY scripts/tinker_eval.py --eval freetext --corpus $D/eval_depth_extrap.jsonl $SPEC --tokenizer $TOK --out $D/res_${TAG}_depth_extrap.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval mmlu_pro --corpus $D/eval_mmlu_pro.jsonl $SPEC --tokenizer $TOK --max-tokens 1024 --out $D/res_${TAG}_mmlu_pro.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval bbh_hard --corpus $D/eval_bbh_hard.jsonl $SPEC --tokenizer $TOK --max-tokens 1024 --out $D/res_${TAG}_bbh_hard.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval gsm8k --corpus data/runs/capability/gsm8k_eval.jsonl $SPEC --tokenizer $TOK --max-tokens 1024 --limit 100 --out $D/res_${TAG}_gsm8k.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
echo "[suite] $TAG done"
