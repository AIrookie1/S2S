#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"
RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/fig3b_hidden_size}"
mkdir -p "$RESULT_DIR"
"$PYTHON_BIN" "$HERE/plot_best.py" \
  --input-dir "${SUMMARY_DIR:-$RESULT_DIR/summaries}" \
  --output-prefix "${OUTPUT_PREFIX:-$RESULT_DIR/mlp_hidden_ablation_mc1}" \
  --num-folds "${NUM_FOLDS:-2}"
