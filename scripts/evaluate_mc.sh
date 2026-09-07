#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FOLD_IDX="${FOLD_IDX:-1}"
NUM_FOLDS="${NUM_FOLDS:-2}"
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.7}"
BETA="${BETA:--0.15}"
DEVICE="${DEVICE:-cuda}"
FEATURE_STEM="${FEATURE_STEM:-${MODEL}_layer${LAYER}}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$PROTOTYPE_DIR/${FEATURE_STEM}_fold${FOLD_IDX}.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-$RESULT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_a${ALPHA}_b${BETA}.json}"

[[ "$FOLD_IDX" =~ ^[0-9]+$ ]] || { echo "FOLD_IDX must be a non-negative integer" >&2; exit 2; }
test -s "$PROTOTYPE_PATH"
EVAL_ARGS=("$MODEL" --prototype_path "$PROTOTYPE_PATH" --layer "$LAYER"
  --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA"
  --output_path "$OUTPUT_PATH" --device "$DEVICE" --save_details)
[[ -n "$MODEL_DIR" ]] && EVAL_ARGS+=(--model_dir "$MODEL_DIR")
[[ "${FEW_SHOT:-0}" == 1 ]] && EVAL_ARGS+=(--few_shot)
[[ -n "${PRESET:-}" ]] && EVAL_ARGS+=(--preset "$PRESET")
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_intervention)

echo "[3/4] MC evaluation | fold=$FOLD_IDX/$NUM_FOLDS | kappa=$KAPPA alpha=$ALPHA beta=$BETA"
CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" evaluate_mc.py "${EVAL_ARGS[@]}" \
  --model_dir /mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct/
test -s "$OUTPUT_PATH"
echo "MC result ready: $OUTPUT_PATH"
