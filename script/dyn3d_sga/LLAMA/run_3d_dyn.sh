#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_3d_extract.sh"
bash "$SCRIPT_DIR/run_3d_train.sh"
bash "$SCRIPT_DIR/run_3d_eval.sh"
echo "3d-dyn pipeline complete."
