#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
LAYER=${LAYER:-14}; GPU=${GPU:-0}; FOLD_IDX=${FOLD_IDX:-0}
source "$SCRIPT_DIR/../../common.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
WEIGHT_KAPPA=10
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_mean.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$JUDGE_DIR/llama_truthfulqa}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_3d.json}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/evaluate_llm_judge_3d.py "$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER" --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH" --weight_kappa "$WEIGHT_KAPPA" ${EXTRA_ARGS:-}
test -s "$OUTPUT_PATH"
