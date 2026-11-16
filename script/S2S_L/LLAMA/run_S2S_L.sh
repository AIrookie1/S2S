#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_S2S_L_extract.sh"
bash "$SCRIPT_DIR/run_S2S_L_train.sh"
bash "$SCRIPT_DIR/run_S2S_L_eval.sh"
echo "S2S_L pipeline complete."
