#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODEL_KEY="${MODEL_KEY:-LLAMA}"; MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
CHECKPOINTS="${CHECKPOINTS:-$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
NUM_QUESTIONS="${NUM_QUESTIONS:-5}"
NUM_VECTORS="${NUM_VECTORS:-32}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$RESULT_ROOT/experiment/fig2/router_weights}"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"
read -r -a checkpoint_array <<< "$CHECKPOINTS"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" experiments/fig2/plot_router_weights.py \
  --checkpoints "${checkpoint_array[@]}" --model_name "$MODEL" --model_dir "$MODEL_DIR" \
  --layer "$LAYER" --num_questions "$NUM_QUESTIONS" --num_vectors "$NUM_VECTORS" \
  --seed "$SEED" --device "$DEVICE" --dtype "$DTYPE" --output_prefix "$OUTPUT_PREFIX"
