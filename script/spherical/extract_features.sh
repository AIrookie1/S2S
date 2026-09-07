#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

OUTPUT_NAME="${OUTPUT_NAME:-${MODEL}_layer${LAYER}.npz}"
echo "[1/4] Extracting hidden states -> $FEATURE_DIR/$OUTPUT_NAME"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" get_activations.py "$MODEL" \
  --layer "$LAYER" --save_dir "$FEATURE_DIR" "${MODEL_DIR_ARGS[@]}"

DEFAULT_OUTPUT="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
if [[ "$OUTPUT_NAME" != "$(basename "$DEFAULT_OUTPUT")" && -f "$DEFAULT_OUTPUT" ]]; then
  mv "$DEFAULT_OUTPUT" "$FEATURE_DIR/$OUTPUT_NAME"
fi
test -s "$FEATURE_DIR/$OUTPUT_NAME"
echo "Feature file ready: $FEATURE_DIR/$OUTPUT_NAME"
