#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="${1:-LLAMA}"
case "${MODEL_KEY^^}" in
  LLAMA|QWEN) exec bash "$SCRIPT_DIR/${MODEL_KEY^^}/run_llm_judge.sh" ;;
  *) echo "usage: $0 [LLAMA|QWEN]" >&2; exit 2 ;;
esac
