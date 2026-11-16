#!/usr/bin/env bash
set -euo pipefail
FAMILY="${MODEL_FAMILY:-llama}"
case "$FAMILY" in
  llama) WEIGHTS="${MODEL_DIR:-${LLAMA_MODEL_DIR:?Set MODEL_DIR or LLAMA_MODEL_DIR}}" ;;
  qwen) WEIGHTS="${MODEL_DIR:-${QWEN_MODEL_DIR:?Set MODEL_DIR or QWEN_MODEL_DIR}}" ;;
  *) echo "MODEL_FAMILY must be llama or qwen" >&2; exit 2 ;;
esac
exec "${PYTHON_BIN:-python}" -m reproducibility.run generic --model "$FAMILY" --model-dir "$WEIGHTS" "$@"
