#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_generic_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

METHOD="${METHOD:-S2S_R}"
MODEL_FAMILY="${MODEL_FAMILY:-LLAMA}"
MODEL_FAMILY="$(printf '%s' "$MODEL_FAMILY" | tr '[:lower:]' '[:upper:]')"
case "$MODEL_FAMILY" in LLAMA|QWEN) ;; *) echo "MODEL_FAMILY must be LLAMA or QWEN" >&2; exit 2;; esac
case "$METHOD" in
  S2S_L|S2S_L_Static) DIM=dyn2d ;;
  S2S_R|S2S_R_Static) DIM=dyn3d ;;
  *) echo "METHOD must be S2S_L, S2S_L_Static, S2S_R, or S2S_R_Static" >&2; exit 2 ;;
esac
exec bash "$REPO_ROOT/script/$DIM/$MODEL_FAMILY/generic/run_dataset.sh" "$DATASET"
