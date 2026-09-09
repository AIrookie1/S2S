#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/py310/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/py310/bin/python
  else
    PYTHON=python
  fi
fi

SUMMARY_DIR="${SUMMARY_DIR:-$HERE/summaries}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$HERE/fig4a_val_fraction_mc1}"
SEEDS="${SEEDS:-42 123 0}"

export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$HERE/plot_fig4a.py" \
  --input-dir "$SUMMARY_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --seeds "$SEEDS"
