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
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$HERE/fig5_weight_kappa_mc1}"

export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" "$HERE/plot_fig5.py" \
  --input-dir "$SUMMARY_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --num-folds 2
