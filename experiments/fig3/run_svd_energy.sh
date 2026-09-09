#!/usr/bin/env bash
set -euo pipefail

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

FIG3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG3_DIR/../../script/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model_config.sh"

GPU=1

# Experiment grid
SVD_ENERGIES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.9999)
FOLD_IDX=0
NUM_FOLDS=2
POSITIVE_MODE=mean

# SVD geometry and router training: same settings as run_3d_train.sh
VAL_FRACTION=0.5
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03
COSINE_LOSS_WEIGHT=0.003
SEED=42
DEVICE=cuda

# dyn3d evaluation: same settings as run_3d_eval.sh
WEIGHT_KAPPA=10
EVAL_SPLIT=test
DTYPE=float16
MODEL_DIR="${MODEL_DIR:-$EXP_MODEL_DIR}"

FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
CHECKPOINT_DIR="$FIG3_DIR/checkpoints"
DETAIL_DIR="$FIG3_DIR/results"
SUMMARY_JSON="$FIG3_DIR/svd_energy_mc1.json"
mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"

# Start a fresh summary. Detailed result files and checkpoints use unique
# energy-specific names and therefore never overwrite the normal experiment.
rm -f "$SUMMARY_JSON"

for SVD_ENERGY in "${SVD_ENERGIES[@]}"; do
  ENERGY_TAG="${SVD_ENERGY/./p}"
  CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_energy${ENERGY_TAG}.npz"
  EVAL_JSON="$DETAIL_DIR/energy${ENERGY_TAG}_eval.json"

  echo
  echo "========== SVD_ENERGY=$SVD_ENERGY: train =========="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
    --feature_path "$FEATURE_PATH" \
    --positive_mode "$POSITIVE_MODE" \
    --output_path "$CHECKPOINT_PATH" \
    --num_folds "$NUM_FOLDS" \
    --fold_idx "$FOLD_IDX" \
    --val_fraction "$VAL_FRACTION" \
    --energy "$SVD_ENERGY" \
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

  echo "========== SVD_ENERGY=$SVD_ENERGY: evaluate =========="
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

  "$PYTHON_BIN" "$FIG3_DIR/collect_mc1.py" \
    --energy "$SVD_ENERGY" \
    --eval-json "$EVAL_JSON" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$SUMMARY_JSON"
done

echo
echo "Fig. 3 MC1 summary: $SUMMARY_JSON"
