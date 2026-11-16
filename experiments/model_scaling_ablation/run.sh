#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-qwen35}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models}"
case "$MODEL" in
  qwen05) MODEL_NAME=Qwen2.5-0.5B-Instruct ;;
  qwen15) MODEL_NAME=Qwen2.5-1.5B-Instruct ;;
  qwen3) MODEL_NAME=Qwen2.5-3B-Instruct ;;
  qwen14) MODEL_NAME=Qwen2.5-14B-Instruct ;;
  qwen35) MODEL_NAME=Qwen3.5-9B ;;
  qwen32) MODEL_NAME=Qwen2.5-32B-Instruct ;;
  nemotron_nano8b) MODEL_NAME=Llama-3.1-Nemotron-Nano-8B-v1 ;;
  *) echo 'MODEL must be one of: qwen05 qwen15 qwen3 qwen14 qwen35 qwen32 nemotron_nano8b' >&2; exit 2 ;;
esac
cd "$ROOT"
exec "$PYTHON_BIN" "$HERE/run.py" --model "$MODEL" --model-dir "$MODEL_ROOT/$MODEL_NAME" "$@"
