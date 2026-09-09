#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"

NUM_FOLDS="${NUM_FOLDS:-2}"
FEATURE_FILE="${FEATURE_FILE:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
echo "[2/4] Computing $NUM_FOLDS-fold prototypes from $FEATURE_FILE"
test -s "$FEATURE_FILE"
"$PYTHON_BIN" get_prototypes.py --feature_file "$FEATURE_FILE" \
  --num_folds "$NUM_FOLDS" --save_dir "$PROTOTYPE_DIR"
echo "Prototype files ready in: $PROTOTYPE_DIR"
