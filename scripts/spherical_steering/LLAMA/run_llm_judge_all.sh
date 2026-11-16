#!/usr/bin/env bash
set -euo pipefail
SPHERICAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SPHERICAL_SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/configs/spherical_steering/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/scripts/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

for ((fold=0; fold<NUM_FOLDS; fold++)); do FOLD_IDX="$fold" bash "$SPHERICAL_SCRIPT_DIR/evaluate_llm_judge.sh"; done
echo "LLM-judge evaluation complete: $RESULT_DIR/llm_judge"
