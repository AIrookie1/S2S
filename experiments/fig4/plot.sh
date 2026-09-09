#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

FIG4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/py310/bin/python}"
INPUT_JSON="${INPUT_JSON:-$FIG4_DIR/val_fraction_mc1.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$FIG4_DIR/fig4_val_fraction_mc1}"

export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$FIG4_DIR/plot_fig4.py" \
  --input "$INPUT_JSON" \
  --output-prefix "$OUTPUT_PREFIX"
