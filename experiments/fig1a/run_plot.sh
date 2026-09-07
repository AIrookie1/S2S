#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/py310/bin/python}"
if [[ -x "$PYTHON_BIN" ]] && "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import matplotlib  # noqa: F401
PY
then
  "$PYTHON_BIN" plot_fig1a_mc1.py "$@"
elif [[ -x /root/miniconda3/bin/conda ]]; then
  /root/miniconda3/bin/conda run --no-capture-output -n py310 python plot_fig1a_mc1.py "$@"
else
  python3 plot_fig1a_mc1.py "$@"
fi
