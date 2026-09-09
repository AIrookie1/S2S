#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"

FOLD_IDX="${FOLD_IDX:-0}"
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.7}"
BETA="${BETA:--0.15}"
PRESET="${PRESET:-null}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
FEATURE_STEM="${FEATURE_STEM:-${MODEL}_layer${LAYER}}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$PROTOTYPE_DIR/${FEATURE_STEM}_fold${FOLD_IDX}.npz}"

test -s "$PROTOTYPE_PATH"
JUDGE_ARGS=("$MODEL" --prototype_path "$PROTOTYPE_PATH" --layer "$LAYER"
  --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA" --preset "$PRESET"
  --max_new_tokens "$MAX_NEW_TOKENS" --output_dir "$JUDGE_DIR")
[[ -n "$MODEL_DIR" ]] && JUDGE_ARGS+=(--model_dir "$MODEL_DIR")
[[ "${DISABLE_STEERING:-0}" == 1 ]] && JUDGE_ARGS+=(--disable_steering)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && JUDGE_ARGS+=(--no_instruction)
[[ -n "${MAX_SAMPLES:-}" ]] && JUDGE_ARGS+=(--max_samples "$MAX_SAMPLES")

echo "[4/4] LLM judge | fold=$FOLD_IDX | preset=$PRESET"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" evaluate_llm_judge.py "${JUDGE_ARGS[@]}"
echo "LLM judge output directory: $JUDGE_DIR"
