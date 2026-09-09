#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

cd "$(dirname "$0")/../.."
source "$(pwd)/experiments/model_config.sh"

PYTHON=/root/miniconda3/envs/py310/bin/python
MODEL=$EXP_MODEL
MODEL_DIR=$EXP_MODEL_DIR
CHECKPOINT=features/${MODEL}_layer${LAYER}_dynSVD_fold1_mean.npz
GPU=0

NUM_QUESTIONS=5
TOP_K=5
SEED=3
OUTPUT_PREFIX=experiments/fig2a/figures/fig2a_topk_positions

export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/fig2a/plot_topk_positions.py \
  --checkpoint "$CHECKPOINT" \
  --model_name "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --num_questions "$NUM_QUESTIONS" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  --output_prefix "$OUTPUT_PREFIX"
