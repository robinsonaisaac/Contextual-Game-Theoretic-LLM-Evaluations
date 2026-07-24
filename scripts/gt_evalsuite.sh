#!/bin/bash
# usage: gt_evalsuite.sh <tag> <--base-model X | --model-path tinker://...>
set -e
TAG="$1"; shift
MODELSPEC="$@"
TOK="Qwen/Qwen3-4B-Instruct-2507"
PY=.venv-tinker/bin/python
D=data/runs/gametree
$PY scripts/tinker_eval.py --eval gametree --corpus $D/exp_gametree.jsonl $MODELSPEC --tokenizer $TOK --out $D/res_${TAG}_gametree.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval gsm8k --corpus $D/exp_gsm8k.jsonl $MODELSPEC --tokenizer $TOK --max-tokens 1024 --out $D/res_${TAG}_gsm8k.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval mmlu --corpus $D/exp_mmlu.jsonl $MODELSPEC --tokenizer $TOK --max-tokens 768 --out $D/res_${TAG}_mmlu.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
$PY scripts/tinker_eval.py --eval bbh --corpus $D/exp_bbh.jsonl $MODELSPEC --tokenizer $TOK --max-tokens 768 --out $D/res_${TAG}_bbh.json 2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
echo "[suite] $TAG done"
