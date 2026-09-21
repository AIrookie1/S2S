#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/script/common.sh"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/dyn2d_ablation}"
"$PYTHON_BIN" "$HERE/plot.py" \
  --input "$OUTPUT_DIR/dyn2d_ablation_mc1.csv" \
  --output "$OUTPUT_DIR/dyn2d_ablation_mc1_heatmap.png" \
  --pdf "$OUTPUT_DIR/dyn2d_ablation_mc1_heatmap.pdf"
