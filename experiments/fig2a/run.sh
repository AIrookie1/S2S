#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=/root/miniconda3/envs/py310/bin/python
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
CHECKPOINT=features/llama3.1-8B-Instruct_layer14_dynSVD_fold1_mean.npz
GPU=0

NUM_QUESTIONS=5
TOP_K=16
SEED=42
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
