#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/l2s/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/l2s/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_l2s_fold${FOLD_IDX}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/l2s/$MODEL}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_l2s_mc.json}"
mkdir -p "$OUTPUT_DIR"
test -s "$CHECKPOINT_PATH"

ARGS=(--method-dir l2s --entry-file evaluate_mc.py -- "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH" --model_dir "$MODEL_DIR" --layer "$LAYER"
  --eval_split "$EVAL_SPLIT" --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA"
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --preset "$EVAL_PRESET"
  --output_path "$OUTPUT_PATH" --save_details)
[[ "${FEW_SHOT:-0}" == 1 ]] && ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
[[ "${NO_INTERVENTION:-0}" == 1 ]] && ARGS+=(--no_intervention)
echo "[L2S eval] model=$MODEL layer=$LAYER fold=$FOLD_IDX"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py "${ARGS[@]}"
test -s "$OUTPUT_PATH"
