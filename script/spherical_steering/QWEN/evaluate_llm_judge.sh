#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/spherical_steering/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

FOLD_IDX="${FOLD_IDX:-0}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$PROTO_DIR/${MODEL}_layer${LAYER}_fold${FOLD_IDX}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_DIR/llm_judge}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
mkdir -p "$OUTPUT_DIR"; test -s "$PROTOTYPE_PATH"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/spherical_steering/evaluate_llm_judge.py "$MODEL"   --prototype_path "$PROTOTYPE_PATH" --model_dir "$MODEL_DIR" --layer "$LAYER"   --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA" --seed "$SEED"   --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"   --output_dir "$OUTPUT_DIR" ${EXTRA_ARGS:-}
