#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/spherical_steering/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

if [[ ! -s "$FEATURE_PATH" ]]; then
  echo "[ERROR] Missing feature file: $FEATURE_PATH" >&2
  echo "[HINT] Run feature extraction first:" >&2
  echo "       bash $SCRIPT_DIR/extract_features.sh" >&2
  exit 1
fi
"$PYTHON_BIN" method/spherical_steering/get_prototypes.py   --feature_file "$FEATURE_PATH" --num_folds "$NUM_FOLDS" --save_dir "$PROTO_DIR"
for ((fold=0; fold<NUM_FOLDS; fold++)); do test -s "$PROTO_DIR/${MODEL}_layer${LAYER}_fold${fold}.npz"; done
