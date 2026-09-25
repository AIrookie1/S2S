#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/script/common.sh"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/dyn2d_ablation}"
SUMMARY_DIR="${SUMMARY_DIR:-$OUTPUT_DIR/summaries}"
FIGURE_DIR="${FIGURE_DIR:-$OUTPUT_DIR/figures}"
mkdir -p "$FIGURE_DIR"
shopt -s nullglob
FILES=("$SUMMARY_DIR"/*_fold_mean.csv)
(( ${#FILES[@]} > 0 )) || { echo "No fold-mean CSV files found in $SUMMARY_DIR" >&2; exit 1; }
for INPUT in "${FILES[@]}"; do
  BASE="$(basename "$INPUT" _fold_mean.csv)"
  "$PYTHON_BIN" "$HERE/plot.py" \
    --input "$INPUT" \
    --title "Dyn2D Ablation: $BASE (fold0/fold1 mean)" \
    --output "$FIGURE_DIR/${BASE}_dyn2d_ablation_mc1_heatmap.png" \
    --pdf "$FIGURE_DIR/${BASE}_dyn2d_ablation_mc1_heatmap.pdf"
done
