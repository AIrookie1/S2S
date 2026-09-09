#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=Qwen2.5-7B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct
LAYER=19
GPU=2
source "$SCRIPT_DIR/../../common.sh"

# Fold and data
FOLD_IDX=1
NUM_FOLDS=2
POSITIVE_MODE=mean
VAL_FRACTION=0.50

# SVD decomposition
SVD_MODE=energy
SVD_RANK=32
SVD_ENERGY=0.40

# Router training
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
SCALE_LOSS_WEIGHT=0.03
COSINE_LOSS_WEIGHT=0.03

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
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py "${ARGS[@]}"
test -s "$OUTPUT_PATH"
