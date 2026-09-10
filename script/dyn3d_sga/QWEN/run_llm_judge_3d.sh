#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=Qwen2.5-7B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct
LAYER=${LAYER:-19}; GPU=${GPU:-2}; FOLD_IDX=${FOLD_IDX:-0}
source "$SCRIPT_DIR/../../common.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_mean.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$JUDGE_DIR/qwen_truthfulqa}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
MARGIN="${MARGIN:-0.3}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/evaluate_llm_judge_3d.py "$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER" --output_dir "$OUTPUT_DIR" --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH" --margin "$MARGIN" ${EXTRA_ARGS:-}
