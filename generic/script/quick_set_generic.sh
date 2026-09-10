#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a DATASETS <<< "${DATASETS:-boolq copa storycloze mmlu_global winogrande}"
for dataset in "${DATASETS[@]}"; do
  bash "$SCRIPT_DIR/run_generic_dataset.sh" "$dataset"
done
