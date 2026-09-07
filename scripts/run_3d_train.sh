#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FOLD_IDX="${FOLD_IDX:-1}"
NUM_FOLDS="${NUM_FOLDS:-2}"
POSITIVE_MODE="${POSITIVE_MODE:-mean}"
SVD_MODE="${SVD_MODE:-energy}"
SVD_RANK="${SVD_RANK:-${RANK:-32}}"
SVD_ENERGY="${SVD_ENERGY:-${ENERGY:-0.95}}"
VAL_FRACTION="${VAL_FRACTION:-0.50}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SCALE_LOSS_WEIGHT="${SCALE_LOSS_WEIGHT:-0.03}"
COSINE_LOSS_WEIGHT="${COSINE_LOSS_WEIGHT:-0.03}"
USE_DATA_MODE="${USE_DATA_MODE:-2}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"

case "${SVD_MODE,,}" in
  rank)
    [[ "$SVD_RANK" =~ ^[1-9][0-9]*$ ]] || { echo "SVD_RANK must be positive" >&2; exit 2; }
    SVD_ARGS=(--rank "$SVD_RANK") ;;
  energy)
    awk -v e="$SVD_ENERGY" 'BEGIN { exit !(e > 0 && e <= 1) }' || { echo "SVD_ENERGY must be in (0,1]" >&2; exit 2; }
    SVD_ARGS=(--energy "$SVD_ENERGY") ;;
  *) echo "SVD_MODE must be rank or energy" >&2; exit 2 ;;
esac

TRAIN_ARGS=(--feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE"
  --output_path "$OUTPUT_PATH" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX"
  --val_fraction "$VAL_FRACTION" "${SVD_ARGS[@]}" --router_hidden 256
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE"
  --device "$DEVICE" --seed 42 --scale_loss_weight "$SCALE_LOSS_WEIGHT"
  --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
  --cosine_loss_weight "$COSINE_LOSS_WEIGHT")

echo "3d-dyn router training | fold=$FOLD_IDX | data_mode=$USE_DATA_MODE | aux=$AUX_INTERP_COUNT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" model/methods/dynSVD/train_dynSVD.py "${TRAIN_ARGS[@]}"
test -s "$OUTPUT_PATH"
