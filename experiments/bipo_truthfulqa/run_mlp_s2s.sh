#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "${PYTHON_BIN:-/root/miniconda3/envs/py310/bin/python}" experiments/bipo_truthfulqa/run_mlp_s2s.py "$@"
