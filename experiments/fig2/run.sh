#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

cd "$(dirname "$0")/../.."
source "$(pwd)/experiments/model_config.sh"

PYTHON=${PYTHON:-/root/miniconda3/envs/py310/bin/python}
MODEL_DIR=${MODEL_DIR:-$EXP_MODEL_DIR}
GPU=${CUDA_VISIBLE_DEVICES:-0}
CHECKPOINTS=${CHECKPOINTS:-"features/llama3.1-8B-Instruct_layer14_dynSVD_fold1_mean.npz"}
NUM_QUESTIONS=${NUM_QUESTIONS:-5}
NUM_VECTORS=${NUM_VECTORS:-32}
SEED=${SEED:-42}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-./experiments/fig2/figures/fig2_router_weights}
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
# CHECKPOINTS is intentionally shell-word-list based so callers can override
# it with one or more NPZ paths: CHECKPOINTS="features/a.npz features/b.npz".
read -r -a checkpoint_array <<< "$CHECKPOINTS"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/fig2/plot_router_weights.py \
  --checkpoints "${checkpoint_array[@]}" \
  --model_name llama3.1-8B-Instruct \
  --model_dir "$MODEL_DIR" \
  --num_questions "$NUM_QUESTIONS" \
  --num_vectors "$NUM_VECTORS" \
  --seed "$SEED" \
  --output_prefix "$OUTPUT_PREFIX"
