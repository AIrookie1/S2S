#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=Qwen2.5-7B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct
LAYER=19
GPU=3
source "$SCRIPT_DIR/../../common.sh"

FOLD_IDX=1
POSITIVE_MODE="${POSITIVE_MODE:-mean}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
WEIGHT_KAPPA="${WEIGHT_KAPPA:-10}"

# Centrality prior: hidden state / mean direction / residual direction

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$RESULT_DIR/${MODEL}_l${LAYER}_3d_dyn_fold${FOLD_IDX}_${POSITIVE_MODE}_similarity_adaptive.json}"

EVAL_ARGS=(
  --method-dir dyn3d_sga
  --entry-file evaluate_3d_dyn.py
  --
  "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH"
  --layer "$LAYER"
  --eval_split "${EVAL_SPLIT:-test}"
  --weight_kappa "$WEIGHT_KAPPA"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output_path "$OUTPUT_PATH"
  --save_details
)
EVAL_ARGS+=("${MODEL_DIR_ARGS[@]}")
[[ "${FEW_SHOT:-0}" == 1 ]] && EVAL_ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_intervention)

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py "${EVAL_ARGS[@]}"
test -s "$OUTPUT_PATH"
