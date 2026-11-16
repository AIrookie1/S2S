#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_R/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

# Paper Table 3 is intentionally fold 0 only.
FOLD_IDX="${FOLD_IDX:-0}"
GPU="${GPU:-2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/table3_component_ablation/$MODEL/fold${FOLD_IDX}}"

if [[ ! -s "$CHECKPOINT_PATH" ]]; then
  echo "[component-ablation] missing fold-0 S2S-R checkpoint: $CHECKPOINT_PATH" >&2
  echo "Train it first with: FOLD_IDX=0 bash script/S2S_R/LLAMA/run_S2S_R_train.sh" >&2
  exit 1
fi

ARGS=(
  "$MODEL"
  --model_dir "$MODEL_DIR"
  --checkpoint_path "$CHECKPOINT_PATH"
  --layer "$LAYER"
  --expected_fold "$FOLD_IDX"
  --eval_split test
  --single_slerp_t 0.4
  --fixed_weights 0.6 0.3 0.1
  --weight_kappa 3
  --margin -0.7
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --preset "$EVAL_PRESET"
  --output_dir "$OUTPUT_DIR"
  --save_details
)
[[ "${FEW_SHOT:-0}" == 1 ]] && ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
[[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max_samples "$MAX_SAMPLES")

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$HERE/evaluate.py" "${ARGS[@]}" "$@"
