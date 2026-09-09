#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"

NUM_FOLDS="${NUM_FOLDS:-2}"
for ((fold=0; fold<NUM_FOLDS; fold++)); do
  FOLD_IDX="$fold" bash "$SCRIPT_DIR/evaluate_mc.sh"
done
echo "MC evaluation complete for $NUM_FOLDS folds. Results: $RESULT_DIR"
