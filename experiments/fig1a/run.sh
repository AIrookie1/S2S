#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

cd "$(dirname "$0")/../.."
source "$(pwd)/experiments/model_config.sh"

PYTHON=${PYTHON:-/root/miniconda3/envs/py310/bin/python}
MODEL_DIR=${MODEL_DIR:-$EXP_MODEL_DIR}
GPU=${CUDA_VISIBLE_DEVICES:-2}
MODEL_NAME=${MODEL_NAME:-$EXP_MODEL}
FOLDS=${FOLDS:-"0 1"}
LAYERS=${LAYERS:-"$(seq 0 31)"}
ALPHA=${ALPHA:-1.0}
OUTPUT_DIR=${OUTPUT_DIR:-./experiments/fig1a/results}
FEATURE_DIR=${FEATURE_DIR:-./features}
FEATURE_TEMPLATE=${FEATURE_TEMPLATE:-"${FEATURE_DIR}/${MODEL_NAME}_layer{layer}_dynSVD_pairs_fold{fold}.npz"}
# Set EXTRACT_FEATURES=1 to create missing per-layer pair files first. This is
# optional because extraction is expensive; the default consumes existing NPZs.
EXTRACT_FEATURES=${EXTRACT_FEATURES:-1}
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$EXTRACT_FEATURES" == "1" ]]; then
  for layer in $LAYERS; do
    base="${FEATURE_DIR}/${MODEL_NAME}_layer${layer}_dynSVD_pairs.npz"
    if [[ ! -f "$base" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" method/dyn3d_sga/extract_features.py \
        "$MODEL_NAME" --layer "$layer" --save_dir "$FEATURE_DIR" \
        --output_name "$(basename "$base")" --model_dir "$MODEL_DIR" \
        --batch_size 1 --num_folds 2 --split_seed 42 --dtype float16 --device cuda
    fi
  done
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/fig1a/evaluate_all_layers.py \
  "$MODEL_NAME" --feature_template "$FEATURE_TEMPLATE" \
  --layers $LAYERS --folds $FOLDS --model_dir "$MODEL_DIR" \
  --alpha "$ALPHA" --output_dir "$OUTPUT_DIR"
