#!/bin/bash
# Tier-5 generalization battery. Usage:
#   bash scripts/tier5_evalsuite.sh base --base-model Qwen/Qwen3-30B-A3B-Instruct-2507
#   bash scripts/tier5_evalsuite.sh sft  --model-path "$(cat data/runs/tier5/sft_checkpoint.txt)"
#   bash scripts/tier5_evalsuite.sh rl   --model-path "$(cat data/runs/tier5/rl_checkpoint.txt)"
# Add --dry-run as the last argument to print commands without executing them.
set -e
TAG="$1"; shift
case "$TAG" in
    base|sft|rl) ;;
    *) echo "usage: tier5_evalsuite.sh {base|sft|rl} [--base-model ...|--model-path ...] [--dry-run]" >&2
       echo "  (guard: refusing to run with TAG='$TAG' — paid sampling)" >&2
       exit 1 ;;
esac

DRY_RUN=0
SPEC=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        SPEC+=("$arg")
    fi
done

TOK="Qwen/Qwen3-30B-A3B-Instruct-2507"; PY=.venv-tinker/bin/python; D=data/runs/tier5; B=data/runs/bbh

run(){
    local EVAL_KIND="$1" CORPUS="$2" NAME="$3" MAXTOK="${4:-4096}"
    local OUT_FILE="$D/t5_${TAG}_${NAME}.json"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $PY scripts/tinker_eval.py --eval $EVAL_KIND --corpus $CORPUS ${SPEC[*]} --tokenizer $TOK --temperature 0 --max-tokens $MAXTOK --out $OUT_FILE"
    else
        $PY scripts/tinker_eval.py --eval "$EVAL_KIND" --corpus "$CORPUS" \
            "${SPEC[@]}" --tokenizer "$TOK" \
            --temperature 0 --max-tokens "$MAXTOK" --out "$OUT_FILE" \
            2>&1 | grep -vE "PyTorch was not found|HF_TOKEN"
    fi
}

# 1) in-domain extrapolated horizons (long traces — 12k budget per MAXTOK map in tier5_gate.py)
run ledger $D/eval_indomain_trees_h127.jsonl            indom_trees_d7   12000
run ledger $D/eval_indomain_register_machine_h150.jsonl indom_reg_n150   12000
run ledger $D/eval_indomain_graph_search_h130.jsonl     indom_graph_130  12000
run ledger $D/eval_indomain_forward_chain_h130.jsonl    indom_chain_130  12000

# 2) held-out families (zero-shot)
run ledger $D/eval_heldout_object_tracking.jsonl        heldout_tracking 4096
run ledger $D/eval_heldout_scheduling.jsonl             heldout_scheduling 4096

# 3) real benchmarks (>=3) + short-form control
run ledger $B/multistep_arithmetic_two_eval.jsonl               bbh_arith    2048
run ledger $B/tracking_shuffled_objects_three_objects_eval.jsonl bbh_track    2048
run ledger $B/dyck_languages_eval.jsonl                         bbh_dyck     2048
run gsm8k  data/runs/capability/gsm8k_eval.jsonl                gsm8k_control 1536

echo "[tier5-suite] $TAG done"
