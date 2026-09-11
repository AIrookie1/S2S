#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/spherical_steering/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/spherical_steering/get_activations.py "$MODEL"   --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$(dirname "$FEATURE_PATH")"
test -s "$FEATURE_PATH"
