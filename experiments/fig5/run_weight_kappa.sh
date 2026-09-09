#!/usr/bin/env bash
set -euo pipefail
FIG5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG5_DIR/../../script/common.sh"
GPU=0
WEIGHT_KAPPAS=(1e5 1e6 1e7 1e8)
FOLD_IDX=0; NUM_FOLDS=2; POSITIVE_MODE=mean
ROUTER_HIDDEN=256; EPOCHS=2000; BATCH_SIZE=256; LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03; COSINE_LOSS_WEIGHT=0.003; SEED=42; DEVICE=cuda
MODEL_DIR="${MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct}"
FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
CHECKPOINT_DIR="$FIG5_DIR/checkpoints"; DETAIL_DIR="$FIG5_DIR/results"; SUMMARY_JSON="$FIG5_DIR/weight_kappa_mc1.json"
mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"; rm -f "$SUMMARY_JSON"
for WEIGHT_KAPPA in "${WEIGHT_KAPPAS[@]}"; do
  TAG="${WEIGHT_KAPPA/./p}"; CHECKPOINT="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_kappa${TAG}.npz"; EVAL_JSON="$DETAIL_DIR/kappa${TAG}_eval.json"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py --feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" --output_path "$CHECKPOINT" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED" --scale_loss_weight "$SCALE_LOSS_WEIGHT" --cosine_loss_weight "$COSINE_LOSS_WEIGHT" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
  "$PYTHON_BIN" "$FIG5_DIR/collect_mc1.py" --kappa "$WEIGHT_KAPPA" --eval-json "$EVAL_JSON" --checkpoint "$CHECKPOINT" --output "$SUMMARY_JSON"
done
echo "Fig. 5 MC1 summary: $SUMMARY_JSON"
