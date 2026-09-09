#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
LAYER=${LAYER:-14}; GPU=${GPU:-3}; FOLD_IDX=${FOLD_IDX:-0}
source "$SCRIPT_DIR/../../common.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
WEIGHT_KAPPA=1e2
WEIGHT_TEMPERATURE=1
HIDDEN_PRIOR=0.33
MEAN_PRIOR=0.33
RESIDUAL_PRIOR=0.34
USE_GATE=0
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_mean.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$JUDGE_DIR/llama_truthfulqa}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_3d.json}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/evaluate_llm_judge_3d.py "$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER" --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --weight_kappa "$WEIGHT_KAPPA" --weight_temperature "$WEIGHT_TEMPERATURE" --hidden_prior "$HIDDEN_PRIOR" --mean_prior "$MEAN_PRIOR" --residual_prior "$RESIDUAL_PRIOR" --use_gate "$USE_GATE" --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH" ${EXTRA_ARGS:-}
