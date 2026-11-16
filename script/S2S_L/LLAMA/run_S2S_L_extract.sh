#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_L/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-${MODEL}_layer${LAYER}.npz}"
echo "[S2S_L extract] model=$MODEL layer=$LAYER"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/S2S_L/extract_features.py "$MODEL"   --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$FEATURE_DIR"   --output_name "$OUTPUT_NAME" --batch_size "$EXTRACT_BATCH_SIZE"   --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
test -s "$FEATURE_DIR/$OUTPUT_NAME"
