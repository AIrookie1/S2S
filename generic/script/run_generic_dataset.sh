#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_generic_dataset.sh <boolq|copa|storycloze|mmlu_global|winogrande>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Historical compatibility wrapper.  The maintained launchers/configs live
# under script/dyn3d_sga/<family>/generic and use the direct-vector MLP.
MODEL_FAMILY="${MODEL_FAMILY:-LLAMA}"
MODEL_FAMILY="$(printf '%s' "$MODEL_FAMILY" | tr '[:lower:]' '[:upper:]')"
case "$MODEL_FAMILY" in
  LLAMA|QWEN) ;;
  *)
    echo "MODEL_FAMILY must be LLAMA or QWEN (got: $MODEL_FAMILY)" >&2
    exit 2
    ;;
esac

exec bash "$REPO_ROOT/script/dyn3d_sga/$MODEL_FAMILY/generic/run_dataset.sh" "$DATASET"
