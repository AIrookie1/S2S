#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source "$(pwd)/experiments/model_config.sh"

PYTHON=/root/miniconda3/envs/py310/bin/python

LAYER=14
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct

# ============================================================
# Fold setup
#
# fold0 router -> evaluate ONLY fold1 questions
# fold1 router -> evaluate ONLY fold0 questions
# ============================================================
ROUTER_FOLD=0
EVAL_FOLD=1


# ============================================================
# Stage 1: per-question Top-K
#
# For every held-out question, select the K router indices
# with the largest router weights.
#
# Example:
#   TOPK=30
#
# means every question contributes exactly 30 router indices.
# ============================================================
TOPK="${TOPK:-10}"


# ============================================================
# Stage 2: global Top-K
#
# After accumulating frequencies over ALL held-out questions,
# keep only the globally most frequent router indices.
#
# Default:
#   DISPLAY_TOPK=30
#
# Therefore the final figure contains exactly 30 bars.
# ============================================================
DISPLAY_TOPK="${DISPLAY_TOPK:-30}"


# ============================================================
# Input files
# ============================================================
FEATURE_PATH=features/${MODEL}_layer${LAYER}.npz

CHECKPOINT=features/${MODEL}_layer${LAYER}_dynSVD_fold${ROUTER_FOLD}_mean.npz


# ============================================================
# Runtime
# ============================================================
GPU=0
SEED=3


# ============================================================
# Output
#
# Example:
#
# fig2b_qtop30_globaltop30.png
# fig2b_qtop30_globaltop30.pdf
# fig2b_qtop30_globaltop30.svg
# fig2b_qtop30_globaltop30_frequencies.csv
# fig2b_qtop30_globaltop30_questions.csv
# fig2b_qtop30_globaltop30.json
# ============================================================
OUTPUT_PREFIX=experiments/fig2b/figures/fig2b_qtop${TOPK}_globaltop${DISPLAY_TOPK}


export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$(dirname "$OUTPUT_PREFIX")"


CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON" experiments/fig2b/plot_peak_index_counts.py \
  --checkpoint "$CHECKPOINT" \
  --feature_path "$FEATURE_PATH" \
  --model_name "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --layer "$LAYER" \
  --router_fold "$ROUTER_FOLD" \
  --eval_fold "$EVAL_FOLD" \
  --topk "$TOPK" \
  --display_topk "$DISPLAY_TOPK" \
  --seed "$SEED" \
  --output_prefix "$OUTPUT_PREFIX"
