#!/usr/bin/env bash
set -euo pipefail

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
LAYER=14
GPU=0
source "$SCRIPT_DIR/../../common.sh"

# Fold and data
FOLD_IDX=1
NUM_FOLDS=2
POSITIVE_MODE=mean
VAL_FRACTION=0.5

# SVD decomposition
SVD_MODE=rank
SVD_RANK=300
SVD_ENERGY=0.90

# Router training
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03
COSINE_LOSS_WEIGHT=0.003

# Runtime and paths
DEVICE=cuda
SEED=42
FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
OUTPUT_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz"

if [[ "$SVD_MODE" == rank ]]; then
  SVD_ARGS=(--rank "$SVD_RANK")
else
  SVD_ARGS=(--energy "$SVD_ENERGY")
fi
ARGS=(--feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE"
  --output_path "$OUTPUT_PATH" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX"
  --val_fraction "$VAL_FRACTION" "${SVD_ARGS[@]}" --router_hidden "$ROUTER_HIDDEN"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE"
  --device "$DEVICE" --seed "$SEED" --scale_loss_weight "$SCALE_LOSS_WEIGHT"
  --cosine_loss_weight "$COSINE_LOSS_WEIGHT")

echo "[train] Similarity-Graph Adaptive 3D router | fold=$FOLD_IDX | svd=$SVD_MODE"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py "${ARGS[@]}" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
test -s "$OUTPUT_PATH"
