#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/configs/S2S_R/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/scripts/common.sh"

GPU="${GPU:-1}"
METHOD="${METHOD:-S2S_R}"
[[ "$METHOD" == "S2S_R" || "$METHOD" == "S2S_R_Static" ]] || { echo "S2S_R METHOD must be S2S_R or S2S_R_Static" >&2; exit 2; }
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/$METHOD/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_${METHOD}_mc.json}"
mkdir -p "$OUTPUT_DIR"
test -s "$CHECKPOINT_PATH"
EVAL_ARGS=(--method-dir S2S_R --entry-file evaluate_S2S_R.py -- "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH" --model_dir "$MODEL_DIR" --layer "$LAYER"
  --eval_split "$EVAL_SPLIT" --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN" --method "$METHOD"
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --preset "$EVAL_PRESET"
  --output_path "$OUTPUT_PATH" --save_details)
[[ "${FEW_SHOT:-0}" == 1 ]] && EVAL_ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && EVAL_ARGS+=(--no_intervention)
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_method.py "${EVAL_ARGS[@]}"
test -s "$OUTPUT_PATH"
