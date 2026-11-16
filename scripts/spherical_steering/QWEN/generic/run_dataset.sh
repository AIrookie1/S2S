#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: run_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG="$REPO_ROOT/configs/spherical_steering/QWEN/generic/$DATASET.sh"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
source "$CONFIG"; source "$REPO_ROOT/scripts/common.sh"
GEN_FEATURE_DIR="${GEN_FEATURE_DIR:-$FEATURE_DIR/generic/spherical_steering/$MODEL}"
GEN_PROTO_DIR="${GEN_PROTO_DIR:-$PROTOTYPE_DIR/generic/spherical_steering/$MODEL}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/generic/$DATASET/spherical_steering}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"; RUN_PROTOTYPE="${RUN_PROTOTYPE:-1}"; RUN_EVAL="${RUN_EVAL:-1}"; DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$GEN_FEATURE_DIR" "$GEN_PROTO_DIR" "$RESULT_DIR"
run_cmd(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '
'; else "$@"; fi; }
require_file(){ [[ "$DRY_RUN" == 1 || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }
FEATURE_PATH="$GEN_FEATURE_DIR/${MODEL}_${DATASET}_train_l${LAYER}.npz"
PROTOTYPE_PATH="$GEN_PROTO_DIR/${MODEL}_${DATASET}_train_l${LAYER}_proto.npz"
OUTPUT_PATH="$RESULT_DIR/${MODEL}_${DATASET}_mc1.json"
if [[ "$RUN_EXTRACT" == 1 ]]; then run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/spherical_steering/generic/get_activations_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --split train --layer "$LAYER" --save_dir "$GEN_FEATURE_DIR" --seed "$SEED"; fi
if [[ "$RUN_PROTOTYPE" == 1 ]]; then require_file "$FEATURE_PATH"; run_cmd "$PYTHON_BIN" methods/spherical_steering/generic/get_prototypes_generic.py --feature_file "$FEATURE_PATH" --save_dir "$GEN_PROTO_DIR"; fi
if [[ "$RUN_EVAL" == 1 ]]; then require_file "$PROTOTYPE_PATH"; run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/spherical_steering/generic/evaluate_generic.py --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --layer "$LAYER" --prototype_path "$PROTOTYPE_PATH" --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA" --seed "$SEED" --dtype "$DTYPE" --output_path "$OUTPUT_PATH"; fi
