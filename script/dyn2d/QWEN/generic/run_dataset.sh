#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: run_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$REPO_ROOT/hyperparams/dyn2d/QWEN/generic/$DATASET.sh"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
source "$CONFIG"; source "$REPO_ROOT/script/common.sh"
GEN_FEATURE_DIR="${GEN_FEATURE_DIR:-$FEATURE_DIR/generic/dyn2d/$MODEL}"
GEN_CHECKPOINT_DIR="${GEN_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/generic/dyn2d/$MODEL}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/generic/$DATASET/dyn2d}"
RUN_EXTRACT="${RUN_EXTRACT:-0}"; RUN_TRAIN="${RUN_TRAIN:-1}"; RUN_EVAL="${RUN_EVAL:-1}"; DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$GEN_FEATURE_DIR" "$GEN_CHECKPOINT_DIR" "$RESULT_DIR"
run_cmd(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
require_file(){ [[ "$DRY_RUN" == 1 || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }
FEATURE_PATH="$GEN_FEATURE_DIR/${MODEL}_${DATASET}_train_l${LAYER}.npz"
CHECKPOINT_PATH="$GEN_CHECKPOINT_DIR/${MODEL}_${DATASET}_l${LAYER}_dyn2d.npz"
OUTPUT_PATH="$RESULT_DIR/${MODEL}_${DATASET}_mc1.json"
if [[ "$RUN_EXTRACT" == 1 ]]; then run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/generic/get_activations_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --split train --layer "$LAYER" --save_dir "$GEN_FEATURE_DIR" --seed "$SEED" --dtype "$DTYPE"; fi
if [[ "$RUN_TRAIN" == 1 ]]; then require_file "$FEATURE_PATH"; run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/train_dyn_mlp.py --feature_path "$FEATURE_PATH" --output_path "$CHECKPOINT_PATH" --positive_mode "$POSITIVE_MODE" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED"; fi
if [[ "$RUN_EVAL" == 1 ]]; then require_file "$CHECKPOINT_PATH"; run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/generic/evaluate_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --layer "$LAYER" --checkpoint_path "$CHECKPOINT_PATH" --strength "$STRENGTH" --slerp_t "$SLERP_T" --seed "$SEED" --dtype "$DTYPE" --output_path "$OUTPUT_PATH"; fi
