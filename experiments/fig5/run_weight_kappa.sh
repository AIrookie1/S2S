#!/usr/bin/env bash
set -euo pipefail

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
FIG5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG5_DIR/../../script/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model_config.sh"
GPU=${GPU:-2}; WEIGHT_KAPPAS=(0.1 1 5 10 15 20 100)
FOLD_IDX=0; NUM_FOLDS=2; POSITIVE_MODE=mean
ROUTER_HIDDEN=256; EPOCHS=2000; BATCH_SIZE=256; LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03; COSINE_LOSS_WEIGHT=0.003; SEED=42; DEVICE=cuda
VAL_FRACTION=${VAL_FRACTION:-0.5}; SVD_RANK=${SVD_RANK:-40};
EVAL_SPLIT=${EVAL_SPLIT:-test}; DTYPE=${DTYPE:-float16};
MODEL_DIR="${MODEL_DIR:-$EXP_MODEL_DIR}"
FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
CHECKPOINT_DIR="$FIG5_DIR/checkpoints"; DETAIL_DIR="$FIG5_DIR/results"; SUMMARY_JSON="$FIG5_DIR/weight_kappa_mc1.json"
mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"; rm -f "$SUMMARY_JSON"
for WEIGHT_KAPPA in "${WEIGHT_KAPPAS[@]}"; do
  TAG="${WEIGHT_KAPPA/./p}"; CHECKPOINT="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_kappa${TAG}.npz"; EVAL_JSON="$DETAIL_DIR/kappa${TAG}_eval.json"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py --feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" --output_path "$CHECKPOINT" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" --rank "$SVD_RANK" --router_hidden "$ROUTER_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED" --scale_loss_weight "$SCALE_LOSS_WEIGHT" --cosine_loss_weight "$COSINE_LOSS_WEIGHT" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
  test -s "$CHECKPOINT"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py --method-dir dyn3d_sga --entry-file evaluate_3d_dyn.py -- "$MODEL" --checkpoint_path "$CHECKPOINT" --model_dir "$MODEL_DIR" --layer "$LAYER" --eval_split "$EVAL_SPLIT" --weight_kappa "$WEIGHT_KAPPA"  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --output_path "$EVAL_JSON" --save_details
  test -s "$EVAL_JSON"
  "$PYTHON_BIN" "$FIG5_DIR/collect_mc1.py" --kappa "$WEIGHT_KAPPA" --eval-json "$EVAL_JSON" --checkpoint "$CHECKPOINT" --output "$SUMMARY_JSON"
done
echo "Fig. 5 MC1 summary: $SUMMARY_JSON"
