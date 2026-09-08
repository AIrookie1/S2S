#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=/root/miniconda3/envs/py310/bin/python
MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
GPU=0

FEATURE_DIR=./features_generic
CHECKPOINT_DIR=./checkpoints_generic
WEIGHT_KAPPA=5.0
WEIGHT_TEMPERATURE=0.1
USE_GATE=0

# Dyn3D training
SVD_ENERGY=0.90
ROUTER_HIDDEN=256
EPOCHS=2000
BATCH_SIZE=256
LEARNING_RATE=0.001
VAL_FRACTION=0.70

mkdir -p "$FEATURE_DIR" "$CHECKPOINT_DIR"

run_dataset() {
  local dataset="$1"
  local layer="$2"
  local feature="$FEATURE_DIR/${MODEL}_${dataset}_train_l${layer}.npz"
  local checkpoint="$CHECKPOINT_DIR/${MODEL}_${dataset}_l${layer}_dyn3d.npz"

  echo "=== $dataset | layer=$layer | extract ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/get_activations_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" \
    --dataset "$dataset" --split train --layer "$layer" --save_dir "$FEATURE_DIR"

  echo "=== $dataset | layer=$layer | train Dyn3D ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
    --feature_path "$feature" --output_path "$checkpoint" \
    --positive_mode mean --num_folds 1 --fold_idx 0 \
    --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" --val_fraction "$VAL_FRACTION" \
    --device cuda

  echo "=== $dataset | layer=$layer | evaluate ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/evaluate_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" \
    --dataset "$dataset" --layer "$layer" --checkpoint_path "$checkpoint" \
    --weight_kappa "$WEIGHT_KAPPA" \
    --weight_temperature "$WEIGHT_TEMPERATURE" --use_gate "$USE_GATE"
}

# Keep the layers and benchmark settings from the original generic pipeline.
run_dataset boolq 12
run_dataset copa 16
run_dataset storycloze 27
run_dataset mmlu_global 28
run_dataset winogrande 16
