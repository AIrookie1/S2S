#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: run_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$REPO_ROOT/hyperparams/dyn3d_sga/QWEN/generic/$DATASET.sh"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
source "$CONFIG"; source "$REPO_ROOT/script/common.sh"
METHOD="${METHOD:-S2S_R}"
[[ "$METHOD" == "S2S_R" || "$METHOD" == "S2S_R_Static" ]] || { echo "dyn3d METHOD must be S2S_R or S2S_R_Static" >&2; exit 2; }
GEN_FEATURE_DIR="${GEN_FEATURE_DIR:-$FEATURE_DIR/generic/dyn3d_sga/$MODEL}"
GEN_CHECKPOINT_DIR="${GEN_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/generic/dyn3d_sga/$MODEL}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/generic/$DATASET/$METHOD}"
RUN_EXTRACT="${RUN_EXTRACT:-0}"; RUN_TRAIN="${RUN_TRAIN:-1}"; RUN_EVAL="${RUN_EVAL:-1}"; DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$GEN_FEATURE_DIR" "$GEN_CHECKPOINT_DIR" "$RESULT_DIR"
run_cmd(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '
'; else "$@"; fi; }
require_file(){ [[ "$DRY_RUN" == 1 || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }
validate_feature_file(){
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
    raise SystemExit(
        f"Invalid feature file: {path}: {exc}. "
        "Regenerate it with RUN_EXTRACT=1 RUN_TRAIN=0 RUN_EVAL=0."
    )
PY
}
FEATURE_PATH="$GEN_FEATURE_DIR/${MODEL}_${DATASET}_train_l${LAYER}.npz"
CHECKPOINT_PATH="$GEN_CHECKPOINT_DIR/${MODEL}_${DATASET}_l${LAYER}_dyn3d.npz"
OUTPUT_PATH="$RESULT_DIR/${MODEL}_${DATASET}_${METHOD}_mc1.json"
if [[ "$RUN_EXTRACT" == 1 ]]; then run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/generic/get_activations_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --split train --layer "$LAYER" --save_dir "$GEN_FEATURE_DIR" --seed "$SEED" --dtype "$DTYPE"; fi
if [[ "$RUN_TRAIN" == 1 ]]; then require_file "$FEATURE_PATH"; validate_feature_file "$FEATURE_PATH"; run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dyn_mlp.py --feature_path "$FEATURE_PATH" --output_path "$CHECKPOINT_PATH" --positive_mode "$POSITIVE_MODE" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED"; fi
if [[ "$RUN_EVAL" == 1 ]]; then require_file "$CHECKPOINT_PATH"; run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/generic/evaluate_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --layer "$LAYER" --checkpoint_path "$CHECKPOINT_PATH" --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN" --method "$METHOD" --seed "$SEED" --dtype "$DTYPE" --output_path "$OUTPUT_PATH"; fi
