#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_generic_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

METHOD="${METHOD:-S2S_R}"
MODEL_FAMILY="${MODEL_FAMILY:-LLAMA}"
MODEL_FAMILY="$(printf '%s' "$MODEL_FAMILY" | tr '[:lower:]' '[:upper:]')"
case "$MODEL_FAMILY" in
  LLAMA|QWEN) ;;
  *) echo "MODEL_FAMILY must be LLAMA or QWEN" >&2; exit 2 ;;
esac

case "$METHOD" in
  S2S_L|S2S_L_Static)
    exec bash "$REPO_ROOT/script/dyn2d/$MODEL_FAMILY/generic/run_dataset.sh" "$DATASET"
    ;;
  S2S_R|S2S_R_Static)
    exec bash "$REPO_ROOT/script/dyn3d/$MODEL_FAMILY/generic/run_dataset.sh" "$DATASET"
    ;;
  L2S_LINEAR|L2S-LINEAR|l2s_linear|l2s-linear)
    exec bash "$REPO_ROOT/script/l2s_linear/$MODEL_FAMILY/generic/run_dataset.sh" "$DATASET"
    ;;
  *)
    echo "METHOD must be S2S_L, S2S_L_Static, S2S_R, S2S_R_Static, or L2S_LINEAR" >&2
    exit 2
    ;;
esac
