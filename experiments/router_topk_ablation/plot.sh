#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/py310/bin/python}"
RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/router_topk_ablation}"
mkdir -p "$RESULT_DIR"
"$PYTHON" "$HERE/plot.py" --input-dir "${SUMMARY_DIR:-$RESULT_DIR/summaries}" --output-prefix "${OUTPUT_PREFIX:-$RESULT_DIR/router_topk_ablation_mc1}" --num-folds 2
