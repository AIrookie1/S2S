#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_2d_extract.sh"
bash "$SCRIPT_DIR/run_2d_train.sh"
bash "$SCRIPT_DIR/run_2d_eval.sh"
echo "dyn2d pipeline complete."
