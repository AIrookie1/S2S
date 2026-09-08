#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/py310/bin/python}"
"$PYTHON" "$HERE/plot_fig5.py" --input "${INPUT_JSON:-$HERE/weight_kappa_mc1.json}" --output-prefix "${OUTPUT_PREFIX:-$HERE/fig5_weight_kappa_mc1}"
