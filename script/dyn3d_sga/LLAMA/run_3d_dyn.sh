#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD_IDX=0

bash "$SCRIPT_DIR/run_3d_extract.sh"
bash "$SCRIPT_DIR/run_3d_train.sh"
bash "$SCRIPT_DIR/run_3d_eval.sh"
echo "3d-dyn pipeline complete; token steer percentage is shown during inference."
