#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_generic_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/hyperparams/generic/$DATASET/config.sh"
[[ -f "$CONFIG" ]] || { echo "Unknown dataset or missing config: $DATASET" >&2; exit 2; }
source "$CONFIG"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/py310/bin/python ]]; then PYTHON_BIN=/root/miniconda3/envs/py310/bin/python; else PYTHON_BIN=python; fi
fi
FEATURE_DIR="${FEATURE_DIR:-$REPO_ROOT/features_generic}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints_generic}"
MC_RESULT_DIR="${MC_RESULT_DIR:-$REPO_ROOT/generic/result}"
RUN_EXTRACT="${RUN_EXTRACT:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$FEATURE_DIR" "$CHECKPOINT_DIR" "$MC_RESULT_DIR"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
require_file() { [[ "$DRY_RUN" == "1" || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }

FEATURE_PATH="$FEATURE_DIR/${MODEL}_${DATASET}_train_l${LAYER}.npz"
CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL}_${DATASET}_l${LAYER}_dyn3d.npz"
MC_RESULT_PATH="$MC_RESULT_DIR/${DATASET}_mc_results.json"

echo "=== $DATASET | layer=$LAYER | baseline: energy=$SVD_ENERGY top_k=$ROUTER_TOP_K load_balance=$LOAD_BALANCE_LOSS_WEIGHT kappa=$WEIGHT_KAPPA margin=$MARGIN ==="

if [[ "$RUN_EXTRACT" == "1" ]]; then
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/get_activations_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" \
    --split train --layer "$LAYER" --save_dir "$FEATURE_DIR" --seed "$SEED" --dtype "$DTYPE"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  require_file "$FEATURE_PATH"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
    --feature_path "$FEATURE_PATH" --output_path "$CHECKPOINT_PATH" \
    --positive_mode "$POSITIVE_MODE" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" \
    --val_fraction "$VAL_FRACTION" --energy "$SVD_ENERGY" \
    --router_hidden "$ROUTER_HIDDEN" --router_top_k "$ROUTER_TOP_K" \
    --vector_loss_weight "$VECTOR_LOSS_WEIGHT" --scale_loss_weight "$SCALE_LOSS_WEIGHT" \
    --cosine_loss_weight "$COSINE_LOSS_WEIGHT" \
    --load_balance_loss_weight "$LOAD_BALANCE_LOSS_WEIGHT" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" --seed "$SEED" --use_data_mode "$USE_DATA_MODE" \
    --aux_interp_count "$AUX_INTERP_COUNT"
  require_file "$CHECKPOINT_PATH"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  require_file "$CHECKPOINT_PATH"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" generic/evaluate_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" \
    --layer "$LAYER" --checkpoint_path "$CHECKPOINT_PATH" \
    --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN" --seed "$SEED" \
    --dtype "$DTYPE" --output_path "$MC_RESULT_PATH"
  require_file "$MC_RESULT_PATH"
fi
