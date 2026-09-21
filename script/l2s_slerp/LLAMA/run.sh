#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_train.sh"
bash "$SCRIPT_DIR/run_eval.sh"
echo "L2S-SLERP LLAMA train/eval pipeline complete."
