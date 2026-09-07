#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NUM_FOLDS="${NUM_FOLDS:-2}"
for ((fold=0; fold<NUM_FOLDS; fold++)); do
  FOLD_IDX="$fold" bash "$SCRIPT_DIR/evaluate_llm_judge.sh"
done
echo "LLM judge evaluation complete for $NUM_FOLDS folds. Results: $JUDGE_DIR"
