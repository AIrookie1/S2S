#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

cd "$(dirname "$0")/../.."
source "$(pwd)/experiments/model_config.sh"

PYTHON=/root/miniconda3/envs/py310/bin/python
LAYER=14
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct

# fold0 router -> ONLY fold1 questions
ROUTER_FOLD=0
EVAL_FOLD=1

# Both files live under features/.
# The checkpoint's test_q_indices is the authoritative held-out fold.
FEATURE_PATH=features/${MODEL}_layer${LAYER}.npz
CHECKPOINT=features/${MODEL}_layer${LAYER}_dynSVD_fold${ROUTER_FOLD}_mean.npz

GPU=0
NUM_QUESTIONS=5
TOP_K=10
SEED=3
OUTPUT_PREFIX=experiments/fig2a/figures/fig2a_topk_positions

export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/fig2a/plot_topk_positions.py \
  --checkpoint "$CHECKPOINT" \
  --feature_path "$FEATURE_PATH" \
  --model_name "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --layer "$LAYER" \
  --router_fold "$ROUTER_FOLD" \
  --eval_fold "$EVAL_FOLD" \
  --num_questions "$NUM_QUESTIONS" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  --output_prefix "$OUTPUT_PREFIX"
