#!/usr/bin/env bash
set -euo pipefail

FIG3A_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG3A_DIR/../../script/common.sh"

GPU=1

# Rank ablation: 10, 20, ..., 100.
SVD_RANKS=(10 20 30 40 50 60 70 80 90 100)
FOLD_IDX=0
NUM_FOLDS=2
POSITIVE_MODE=mean

# SVD geometry and router training: same settings as Fig. 3.
USE_DATA_MODE=0
AUX_INTERP_COUNT=8
VAL_FRACTION=0.50
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03
COSINE_LOSS_WEIGHT=0.003
SEED=42
DEVICE=cuda

# dyn3d evaluation: same settings as Fig. 3.
WEIGHT_KAPPA=1e2
WEIGHT_TEMPERATURE=1.0
HIDDEN_PRIOR=0.33
MEAN_PRIOR=0.33
RESIDUAL_PRIOR=0.34
USE_GATE=0
EVAL_SPLIT=test
DTYPE=float16
MODEL_DIR="${MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct}"

FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
CHECKPOINT_DIR="$FIG3A_DIR/checkpoints"
DETAIL_DIR="$FIG3A_DIR/results"
SUMMARY_JSON="$FIG3A_DIR/svd_rank_mc1.json"
mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"
rm -f "$SUMMARY_JSON"

for SVD_RANK in "${SVD_RANKS[@]}"; do
  CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_rank${SVD_RANK}.npz"
  EVAL_JSON="$DETAIL_DIR/rank${SVD_RANK}_eval.json"

  echo
  echo "========== SVD_RANK=$SVD_RANK: train =========="
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
    --use_data_mode "$USE_DATA_MODE" \
    --aux_interp_count "$AUX_INTERP_COUNT"
  test -s "$CHECKPOINT_PATH"

  echo "========== SVD_RANK=$SVD_RANK: evaluate =========="
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
    --weight_temperature "$WEIGHT_TEMPERATURE" \
    --hidden_prior "$HIDDEN_PRIOR" \
    --mean_prior "$MEAN_PRIOR" \
    --residual_prior "$RESIDUAL_PRIOR" \
    --use_gate "$USE_GATE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --output_path "$EVAL_JSON" \
    --save_details
  test -s "$EVAL_JSON"

  "$PYTHON_BIN" "$FIG3A_DIR/collect_mc1.py" \
    --rank "$SVD_RANK" \
    --eval-json "$EVAL_JSON" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$SUMMARY_JSON"
done

echo
echo "Fig. 3a MC1 summary: $SUMMARY_JSON"
