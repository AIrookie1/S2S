#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
MARGIN="${MARGIN:-0.3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
LAYER=14
GPU=0
source "$SCRIPT_DIR/../../common.sh"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_NAME="${OUTPUT_NAME:-${MODEL}_layer${LAYER}.npz}"
EXTRACT_ARGS=("$MODEL" --layer "$LAYER" --save_dir "$FEATURE_DIR"
  --output_name "$OUTPUT_NAME" --batch_size "$BATCH_SIZE"
  --device "$DEVICE" --dtype "$DTYPE" "${MODEL_DIR_ARGS[@]}")

echo "3d-dyn feature extraction | model=$MODEL layer=$LAYER device=$DEVICE"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "${EXTRACT_ARGS[@]}"
test -s "$FEATURE_DIR/$OUTPUT_NAME"
