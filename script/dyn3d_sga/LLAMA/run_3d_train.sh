#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/truthfulqa/config.sh"
source "$REPO_ROOT/hyperparams/models.sh"
configure_model llama
GPU="${GPU:-0}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"
source "$REPO_ROOT/script/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
ARGS=(--feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" --output_path "$OUTPUT_PATH"
  --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION"
  --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" --router_top_k "$ROUTER_TOP_K"
  --vector_loss_weight "$VECTOR_LOSS_WEIGHT" --scale_loss_weight "$SCALE_LOSS_WEIGHT"
  --cosine_loss_weight "$COSINE_LOSS_WEIGHT" --load_balance_loss_weight "$LOAD_BALANCE_LOSS_WEIGHT"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE"
  --device "$DEVICE" --seed "$SEED" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT")
echo "[train] model=$MODEL fold=$FOLD_IDX energy=$SVD_ENERGY top_k=$ROUTER_TOP_K load_balance=$LOAD_BALANCE_LOSS_WEIGHT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py "${ARGS[@]}"
test -s "$OUTPUT_PATH"
