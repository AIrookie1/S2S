#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
FOLD_IDX="${FOLD_IDX:-0}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynMLP_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/component_ablation/$MODEL/fold${FOLD_IDX}_vavg_grid}"
[[ -s "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }
read -r -a KAPPA_GRID <<< "${KAPPAS:-0.1 0.3 0.5 1 2 3 5 10}"
read -r -a MARGIN_GRID <<< "${MARGINS:--0.7 0.34 0.5 0.7 0.9 0.95 0.99 1}"
ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH"
  --layer "$LAYER" --expected_fold "$FOLD_IDX" --eval_split "${EVAL_SPLIT:-test}"
  --variants vavg_aw_gate --vavg_kappas "${KAPPA_GRID[@]}" --vavg_margins "${MARGIN_GRID[@]}"
  --selection_metric "${SELECTION_METRIC:-MC1}" --resume
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --preset "$EVAL_PRESET"
  --output_dir "$OUTPUT_DIR")
[[ "${FEW_SHOT:-0}" == 1 ]] && ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
[[ "${SAVE_DETAILS:-0}" == 1 ]] && ARGS+=(--save_details)
[[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max_samples "$MAX_SAMPLES")
cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -B "$HERE/evaluate.py" "${ARGS[@]}" "$@"
