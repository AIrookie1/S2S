#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_R/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
mkdir -p "$OUTPUT_DIR"
test -s "$FEATURE_PATH"
ARGS=(--feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" --output_path "$OUTPUT_PATH"
  --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION"
  --mlp_hidden "$MLP_HIDDEN"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE"
  --device "$DEVICE" --seed "$SEED")
echo "[train] model=$MODEL fold=$FOLD_IDX layer=$LAYER direct-MLP-hidden=$MLP_HIDDEN"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/S2S_R/train_predictor.py "${ARGS[@]}"
test -s "$OUTPUT_PATH"
