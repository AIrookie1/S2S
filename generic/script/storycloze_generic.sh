#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=/root/miniconda3/envs/py310/bin/python
GPU=0
FEATURE_DIR=./features_generic
CHECKPOINT_DIR=./checkpoints_generic
MC_RESULT_DIR="$SCRIPT_DIR/../result"

mkdir -p "$FEATURE_DIR" "$CHECKPOINT_DIR"

run_dataset() {
  local dataset="$1"
  local layer="$2"

  # Dataset-specific model, training, and evaluation parameters.
  # Add or adjust a dataset block without affecting the others.
  local MODEL MODEL_DIR WEIGHT_KAPPA WEIGHT_TEMPERATURE USE_GATE
  local SVD_ENERGY ROUTER_HIDDEN EPOCHS BATCH_SIZE LEARNING_RATE VAL_FRACTION
  local SCALE_LOSS_WEIGHT COSINE_LOSS_WEIGHT POSITIVE_MODE USE_DATA_MODE AUX_INTERP_COUNT
  local HIDDEN_PRIOR MEAN_PRIOR RESIDUAL_PRIOR
  case "$dataset" in
    storycloze)
      MODEL=llama3.1-8B-Instruct
      MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
      WEIGHT_KAPPA=1e2; WEIGHT_TEMPERATURE=1; USE_GATE=0
      SVD_ENERGY=0.4; ROUTER_HIDDEN=256; EPOCHS=2000; BATCH_SIZE=256
      LEARNING_RATE=0.001; VAL_FRACTION=0.50
      SCALE_LOSS_WEIGHT=0.03; COSINE_LOSS_WEIGHT=0.03
      POSITIVE_MODE=mean; USE_DATA_MODE=0; AUX_INTERP_COUNT=8
      HIDDEN_PRIOR=1; MEAN_PRIOR=1; RESIDUAL_PRIOR=1
      ;;
    *) echo "Unknown dataset: $dataset" >&2; return 2 ;;
  esac

  local feature="$FEATURE_DIR/${MODEL}_${dataset}_train_l${layer}.npz"
  local checkpoint="$CHECKPOINT_DIR/${MODEL}_${dataset}_l${layer}_dyn3d.npz"
  local mc_result="$MC_RESULT_DIR/${dataset}_mc_results.json"

  echo "=== $dataset | layer=$layer | extract ==="
  # Uncomment to regenerate activations when the feature file is missing or outdated.
  # CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/get_activations_generic.py \
  #   --model_name "$MODEL" --model_dir "$MODEL_DIR" \
  #   --dataset "$dataset" --split train --layer "$layer" --save_dir "$FEATURE_DIR"

  echo "=== $dataset | layer=$layer | train Dyn3D ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
    --feature_path "$feature" --output_path "$checkpoint" \
    --positive_mode "$POSITIVE_MODE" --num_folds 1 --fold_idx 0 \
    --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" --val_fraction "$VAL_FRACTION" \
    --scale_loss_weight "$SCALE_LOSS_WEIGHT" --cosine_loss_weight "$COSINE_LOSS_WEIGHT" \
    --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT" \
    --device cuda

  echo "=== $dataset | layer=$layer | evaluate ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/evaluate_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" \
    --dataset "$dataset" --layer "$layer" --checkpoint_path "$checkpoint" \
    --weight_kappa "$WEIGHT_KAPPA" --weight_temperature "$WEIGHT_TEMPERATURE" \
    --use_gate "$USE_GATE" --output_path "$mc_result"
  test -s "$mc_result"
  echo "MC result saved: $mc_result"
}

# run_dataset boolq 12
# run_dataset copa 16
run_dataset storycloze 27
# run_dataset mmlu_global 28
# run_dataset winogrande 16