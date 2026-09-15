#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/l2s/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
FEATURE_PATH_WAS_SET=0
if [[ -n "${FEATURE_PATH+x}" ]]; then FEATURE_PATH_WAS_SET=1; fi
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_pairs.npz}"
if [[ ! -s "$FEATURE_PATH" && "$FEATURE_PATH_WAS_SET" == 0 && -s "$FEATURE_DIR/${MODEL}_layer${LAYER}.npz" ]]; then
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/l2s/truthfulqa/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_layer${LAYER}_l2s_fold${FOLD_IDX}.npz}"
mkdir -p "$OUTPUT_DIR"

if [[ ! -s "$FEATURE_PATH" ]]; then
  if [[ "${RUN_EXTRACT:-0}" != 1 ]]; then
    echo "Missing feature file: $FEATURE_PATH (set RUN_EXTRACT=1 or FEATURE_PATH=...)" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$FEATURE_PATH")"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "$MODEL" \
    --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$(dirname "$FEATURE_PATH")" \
    --output_name "$(basename "$FEATURE_PATH")" --batch_size "$EXTRACT_BATCH_SIZE" \
    --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
fi
test -s "$FEATURE_PATH"

echo "[L2S train] model=$MODEL layer=$LAYER fold=$FOLD_IDX"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/l2s/train_l2s.py \
  --feature_path "$FEATURE_PATH" --output_path "$OUTPUT_PATH" \
  --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" \
  --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED"
test -s "$OUTPUT_PATH"
