#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn3d_sga/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
mkdir -p "$OUTPUT_DIR"
test -s "$FEATURE_PATH"
ARGS=(--feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" --output_path "$OUTPUT_PATH"
  --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION"
  --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" --router_top_k "$ROUTER_TOP_K"
  --load_balance_loss_weight "$LOAD_BALANCE_LOSS_WEIGHT"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE"
  --device "$DEVICE" --seed "$SEED" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT")
echo "[train] model=$MODEL fold=$FOLD_IDX layer=$LAYER"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py "${ARGS[@]}"
test -s "$OUTPUT_PATH"
