#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_FOLDS="${NUM_FOLDS:-2}"
SPLIT_SEED="${SPLIT_SEED:-1}"
OUTPUT_NAME="${OUTPUT_NAME:-${MODEL}_layer${LAYER}_dynSVD_pairs.npz}"

EXTRACT_ARGS=("$MODEL" --layer "$LAYER" --save_dir "$FEATURE_DIR"
  --output_name "$OUTPUT_NAME" --batch_size "$BATCH_SIZE"
  --num_folds "$NUM_FOLDS" --split_seed "$SPLIT_SEED"
  --device "$DEVICE" --dtype "$DTYPE")
[[ -n "$MODEL_DIR" ]] && EXTRACT_ARGS+=(--model_dir "$MODEL_DIR")

echo "3d-dyn feature extraction | model=$MODEL layer=$LAYER device=$DEVICE"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" model/methods/dynSVD/extract_dynSVD_features.py "${EXTRACT_ARGS[@]}" \
  --model_dir /mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct/
test -s "$FEATURE_DIR/$OUTPUT_NAME"
