#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"; MODEL_KEY="${MODEL_KEY:-LLAMA}"; MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"; source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-0}"; ROUTER_FOLD="${ROUTER_FOLD:-0}"; EVAL_FOLD="${EVAL_FOLD:-$((1-ROUTER_FOLD))}"; TOPK="${TOPK:-10}"; DISPLAY_TOPK="${DISPLAY_TOPK:-30}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynSVD_fold${ROUTER_FOLD}_${POSITIVE_MODE}.npz}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$RESULT_ROOT/experiment/router_topk_frequency_analysis/router_topk_frequency_analysis_qtop${TOPK}_globaltop${DISPLAY_TOPK}}"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"; test -s "$CHECKPOINT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$HERE/analyze.py" --checkpoint "$CHECKPOINT" --feature_path "$FEATURE_PATH" --model_name "$MODEL" --model_dir "$MODEL_DIR" --layer "$LAYER" --router_fold "$ROUTER_FOLD" --eval_fold "$EVAL_FOLD" --topk "$TOPK" --display_topk "$DISPLAY_TOPK" --seed "$SEED" --output_prefix "$OUTPUT_PREFIX" ${EXTRA_ARGS:-}
