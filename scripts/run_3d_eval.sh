#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FOLD_IDX="${FOLD_IDX:-0}"
POSITIVE_MODE="${POSITIVE_MODE:-mean}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
WEIGHT_KAPPA="${WEIGHT_KAPPA:-5.0}"
WEIGHT_TEMPERATURE="${WEIGHT_TEMPERATURE:-0.1}"
HIDDEN_COSINE_THRESHOLD="${HIDDEN_COSINE_THRESHOLD:-0.99}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$RESULT_DIR/${MODEL}_l${LAYER}_3d_dyn_fold${FOLD_IDX}_${POSITIVE_MODE}_similarity_adaptive.json}"

EVAL_ARGS=(
  --method-dir 3d-dyn
  --entry-file evaluate_3d_dyn.py
  --
  "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH"
  --layer "$LAYER"
  --eval_split "${EVAL_SPLIT:-test}"
  --weight_kappa "$WEIGHT_KAPPA"
  --weight_temperature "$WEIGHT_TEMPERATURE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output_path "$OUTPUT_PATH"
  --hidden_cosine_threshold "$HIDDEN_COSINE_THRESHOLD"
  --save_details
  --model_dir /mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct
)
[[ -n "${MODEL_DIR:-}" ]] && EVAL_ARGS+=(--model_dir "$MODEL_DIR")
[[ "${FEW_SHOT:-0}" == 1 ]] && EVAL_ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_intervention)

echo "3d-dyn Similarity-Graph adaptive inference | fold=$FOLD_IDX | kappa=$WEIGHT_KAPPA temperature=$WEIGHT_TEMPERATURE"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py "${EVAL_ARGS[@]}"
test -s "$OUTPUT_PATH"
