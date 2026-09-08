#!/usr/bin/env bash
set -euo pipefail

FIG3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/py310/bin/python}"
INPUT_JSON="${INPUT_JSON:-$FIG3_DIR/svd_energy_mc1.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$FIG3_DIR/fig3_svd_energy_mc1}"

export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$FIG3_DIR/plot_fig3.py" \
  --input "$INPUT_JSON" \
  --output-prefix "$OUTPUT_PREFIX"
