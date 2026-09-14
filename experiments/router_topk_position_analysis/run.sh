#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"; MODEL_KEY="${MODEL_KEY:-LLAMA}"; MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"; source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-0}"; ROUTER_FOLD="${ROUTER_FOLD:-0}"; EVAL_FOLD="${EVAL_FOLD:-$((1-ROUTER_FOLD))}"; NUM_QUESTIONS="${NUM_QUESTIONS:-5}"; MIN_UNIQUE_PEAKS="${MIN_UNIQUE_PEAKS:-3}"; TOP_K="${TOP_K:-32}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynSVD_fold${ROUTER_FOLD}_${POSITIVE_MODE}.npz}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$RESULT_ROOT/experiment/router_topk_position_analysis/router_topk_position_analysis}"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"; test -s "$CHECKPOINT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$HERE/analyze.py" --checkpoint "$CHECKPOINT" --feature_path "$FEATURE_PATH" --model_name "$MODEL" --model_dir "$MODEL_DIR" --layer "$LAYER" --router_fold "$ROUTER_FOLD" --eval_fold "$EVAL_FOLD" --num_questions "$NUM_QUESTIONS" --min_unique_peaks "$MIN_UNIQUE_PEAKS" --top_k "$TOP_K" --seed "$SEED" --output_prefix "$OUTPUT_PREFIX" ${EXTRA_ARGS:-}
