#!/usr/bin/env bash
set -euo pipefail

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

FIG4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG4_DIR/../../script/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model_config.sh"

GPU=3

# Experiment grid
VAL_FRACTIONS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FOLD_IDX=0
NUM_FOLDS=2
POSITIVE_MODE=mean

# SVD geometry and router training: only VAL_FRACTION changes.
SVD_RANK=300
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03
COSINE_LOSS_WEIGHT=0.003
SEED=42
DEVICE=cuda

# Same dyn3d evaluation settings as Fig. 3.
WEIGHT_KAPPA=1e3
EVAL_SPLIT=test
DTYPE=float16
MODEL_DIR="${MODEL_DIR:-$EXP_MODEL_DIR}"

FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
CHECKPOINT_DIR="$FIG4_DIR/checkpoints"
DETAIL_DIR="$FIG4_DIR/results"
SUMMARY_JSON="$FIG4_DIR/val_fraction_mc1.json"
mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"
rm -f "$SUMMARY_JSON"

for VAL_FRACTION in "${VAL_FRACTIONS[@]}"; do
  FRACTION_TAG="${VAL_FRACTION/./p}"
  CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_val${FRACTION_TAG}.npz"
  EVAL_JSON="$DETAIL_DIR/val${FRACTION_TAG}_eval.json"

  echo
  echo "========== VAL_FRACTION=$VAL_FRACTION: train =========="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
    --feature_path "$FEATURE_PATH" \
    --positive_mode "$POSITIVE_MODE" \
    --output_path "$CHECKPOINT_PATH" \
    --num_folds "$NUM_FOLDS" \
    --fold_idx "$FOLD_IDX" \
    --val_fraction "$VAL_FRACTION" \
    --rank "$SVD_RANK" \
    --router_hidden "$ROUTER_HIDDEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --scale_loss_weight "$SCALE_LOSS_WEIGHT" \
    --cosine_loss_weight "$COSINE_LOSS_WEIGHT" \
    --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
  test -s "$CHECKPOINT_PATH"

  echo "========== VAL_FRACTION=$VAL_FRACTION: evaluate =========="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py \
    --method-dir dyn3d_sga \
    --entry-file evaluate_3d_dyn.py \
    -- \
    "$MODEL" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_dir "$MODEL_DIR" \
    --layer "$LAYER" \
    --eval_split "$EVAL_SPLIT" \
    --weight_kappa "$WEIGHT_KAPPA" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --output_path "$EVAL_JSON" \
    --save_details
  test -s "$EVAL_JSON"

  "$PYTHON_BIN" "$FIG4_DIR/collect_mc1.py" \
    --val-fraction "$VAL_FRACTION" \
    --eval-json "$EVAL_JSON" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$SUMMARY_JSON"
done

echo
echo "Fig. 4 MC1 summary: $SUMMARY_JSON"
