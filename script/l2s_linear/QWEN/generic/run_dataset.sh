#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$REPO_ROOT/hyperparams/l2s_linear/QWEN/generic/$DATASET.sh"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
source "$CONFIG"
source "$REPO_ROOT/script/common.sh"

METHOD="${METHOD:-L2S_LINEAR}"
case "$METHOD" in
  L2S_LINEAR|L2S-LINEAR|l2s_linear|l2s-linear) ;;
  *) echo "L2S generic METHOD must be L2S_LINEAR (or L2S-LINEAR)" >&2; exit 2 ;;
esac

# Generic answer-level features are method-agnostic. Reuse the canonical
# Dyn3D generic feature directory by default so L2S and Dyn3D see identical
# hidden states; set GEN_FEATURE_DIR to isolate them if desired.
GEN_FEATURE_DIR="${GEN_FEATURE_DIR:-$FEATURE_DIR/generic/dyn3d_sga/$MODEL}"
GEN_CHECKPOINT_DIR="${GEN_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/generic/l2s_linear/$MODEL}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/generic/$DATASET/l2s_linear}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$RUN_EXTRACT" in 0|1|auto) ;; *) echo "RUN_EXTRACT must be 0, 1, or auto" >&2; exit 2 ;; esac
case "$RUN_TRAIN" in 0|1) ;; *) echo "RUN_TRAIN must be 0 or 1" >&2; exit 2 ;; esac
case "$RUN_EVAL" in 0|1) ;; *) echo "RUN_EVAL must be 0 or 1" >&2; exit 2 ;; esac

mkdir -p "$GEN_FEATURE_DIR" "$GEN_CHECKPOINT_DIR" "$RESULT_DIR"

run_cmd() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_file() {
  [[ "$DRY_RUN" == 1 || -s "$1" ]] || {
    echo "Missing required file: $1" >&2
    return 1
  }
}

validate_feature_file() {
  [[ "$DRY_RUN" == 1 ]] && return 0
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
try:
    with np.load(path, allow_pickle=False) as data:
        required = {"activations", "labels", "q_indices"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        activations = data["activations"]
        labels = data["labels"].reshape(-1)
        q_indices = data["q_indices"].reshape(-1)
        if activations.ndim != 2 or len(activations) == 0:
            raise ValueError(f"activations must be non-empty [N, D], got {activations.shape}")
        if not (len(activations) == len(labels) == len(q_indices)):
            raise ValueError("activations, labels, and q_indices are not aligned")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("labels do not contain both 0 and 1")
except Exception as exc:
    raise SystemExit(f"Invalid generic feature file: {path}: {exc}")
PY
}

FEATURE_PATH="$GEN_FEATURE_DIR/${MODEL}_${DATASET}_train_l${LAYER}.npz"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$GEN_CHECKPOINT_DIR/${MODEL}_${DATASET}_l${LAYER}_l2s_linear_fold${FOLD_IDX}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$RESULT_DIR/${MODEL}_${DATASET}_l2s_linear_mc1.json}"

should_extract=0
if [[ "$RUN_EXTRACT" == 1 ]]; then
  should_extract=1
elif [[ "$RUN_EXTRACT" == auto && ! -s "$FEATURE_PATH" ]]; then
  should_extract=1
fi

if [[ "$should_extract" == 1 ]]; then
  run_cmd env \
    EXTRACT_BATCH_SIZE="$EXTRACT_BATCH_SIZE" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" method/l2s_linear/generic/get_activations_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" \
    --split train --layer "$LAYER" --save_dir "$GEN_FEATURE_DIR" \
    --seed "$SEED" --dtype "$DTYPE"
fi

if [[ "$RUN_TRAIN" == 1 ]]; then
  require_file "$FEATURE_PATH"
  validate_feature_file "$FEATURE_PATH"
  echo "[L2S-LINEAR generic train] family=QWEN dataset=$DATASET model=$MODEL layer=$LAYER fold=$FOLD_IDX"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" method/l2s_linear/train_l2s.py \
    --feature_path "$FEATURE_PATH" --output_path "$CHECKPOINT_PATH" \
    --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" \
    --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  require_file "$CHECKPOINT_PATH"
  echo "[L2S-LINEAR generic eval] family=QWEN dataset=$DATASET model=$MODEL layer=$LAYER alpha=$ALPHA"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" method/l2s_linear/generic/evaluate_generic.py \
    --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" \
    --layer "$LAYER" --checkpoint_path "$CHECKPOINT_PATH" --alpha "$ALPHA" \
    --seed "$SEED" --device "$DEVICE" --dtype "$DTYPE" --output_path "$OUTPUT_PATH"
fi
