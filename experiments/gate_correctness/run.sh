#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/py310/bin/python}"
GPU="${GPU:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/experiment/gate_correctness/llama3.1-8B-Instruct}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" "$HERE/evaluate.py" --output_dir "$OUTPUT_DIR" "$@"
