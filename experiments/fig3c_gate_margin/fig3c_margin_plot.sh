#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python}"
RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/fig3c_gate_margin}"
mkdir -p "$RESULT_DIR"
"$PYTHON" "$HERE/fig3c_margin_plot.py" --input-dir "${SUMMARY_DIR:-$RESULT_DIR/summaries}" --output-prefix "${OUTPUT_PREFIX:-$RESULT_DIR/margin_ablation_mc1}" --num-folds 2
