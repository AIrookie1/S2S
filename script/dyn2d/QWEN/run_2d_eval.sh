#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn2d/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/dyn2d/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynMLP_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/dyn2d/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_strength${STRENGTH}_slerp${SLERP_T}_mc.json}"
mkdir -p "$OUTPUT_DIR"
test -s "$CHECKPOINT_PATH"
EVAL_ARGS=(--method-dir dyn2d --entry-file evaluate_2d_dyn.py -- "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH" --model_dir "$MODEL_DIR" --layer "$LAYER"
  --eval_split "$EVAL_SPLIT" --strength "$STRENGTH" --slerp_t "$SLERP_T"
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --preset "$EVAL_PRESET"
  --output_path "$OUTPUT_PATH" --save_details)
[[ "${FEW_SHOT:-0}" == 1 ]] && EVAL_ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_intervention)
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py "${EVAL_ARGS[@]}"
test -s "$OUTPUT_PATH"
