#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/configs/spherical_steering/QWEN/truthfulqa.sh"
source "$REPO_ROOT/scripts/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/spherical_steering/get_activations.py "$MODEL"   --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$(dirname "$FEATURE_PATH")"
if [[ ! -s "$FEATURE_PATH" ]]; then
  echo "[ERROR] Feature extraction finished without creating: $FEATURE_PATH" >&2
  exit 1
fi
echo "[OK] Features: $FEATURE_PATH"
