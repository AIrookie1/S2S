#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NUM_FOLDS="${NUM_FOLDS:-2}"
for ((fold=0; fold<NUM_FOLDS; fold++)); do
  FOLD_IDX="$fold" bash "$SCRIPT_DIR/evaluate_mc.sh"
done
echo "MC evaluation complete for $NUM_FOLDS folds. Results: $RESULT_DIR"
