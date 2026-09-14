#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"; source "$REPO_ROOT/script/common.sh"
MODEL_NAME="${MODEL_NAME:-llama3.1-8B-Instruct}"; RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/experiment/steering_vector_ablation}"
"$PYTHON_BIN" "$HERE/plot.py" --input_dir "$RESULT_DIR" --model_name "$MODEL_NAME" --output_prefix "${OUTPUT_PREFIX:-$RESULT_DIR/steering_vector_ablation_mc1}"
