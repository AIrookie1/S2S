#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/truthfulqa/config.sh"
source "$REPO_ROOT/hyperparams/models.sh"
configure_model llama
GPU="${GPU:-0}"
source "$REPO_ROOT/script/common.sh"
OUTPUT_NAME="${OUTPUT_NAME:-${MODEL}_layer${LAYER}.npz}"
echo "3d-dyn feature extraction | model=$MODEL layer=$LAYER device=$DEVICE dtype=$DTYPE"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "$MODEL"   --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$FEATURE_DIR"   --output_name "$OUTPUT_NAME" --batch_size "$EXTRACT_BATCH_SIZE"   --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
test -s "$FEATURE_DIR/$OUTPUT_NAME"
