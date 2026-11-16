#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_R/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

# Paper Table 4 uses the S2S fold-0 held-out questions.
FOLD_IDX=0
GPU="${GPU:-0}"
S2SL_CHECKPOINT="${S2SL_CHECKPOINT:-$CHECKPOINT_ROOT/S2S_L/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
S2SR_CHECKPOINT="${S2SR_CHECKPOINT:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"

# Important: the repository's two fold builders use opposite conventions.
# Dyn/S2S fold 0 is trained on the first half and held out on the second half;
# spherical fold 1 is trained on the first half and held out on the second half.
SPHERICAL_FOLD_IDX="${SPHERICAL_FOLD_IDX:-1}"
SPHERICAL_PROTOTYPE="${SPHERICAL_PROTOTYPE:-$PROTOTYPE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}_fold${SPHERICAL_FOLD_IDX}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/table4_generation_efficiency/$MODEL/fold${FOLD_IDX}}"

for path in "$S2SL_CHECKPOINT" "$S2SR_CHECKPOINT" "$SPHERICAL_PROTOTYPE"; do
  if [[ ! -s "$path" ]]; then
    echo "[generation-efficiency] missing required file: $path" >&2
    exit 1
  fi
done

ARGS=(
  "$MODEL"
  --model_dir "$MODEL_DIR"
  --s2sl_checkpoint "$S2SL_CHECKPOINT"
  --s2sr_checkpoint "$S2SR_CHECKPOINT"
  --spherical_prototype "$SPHERICAL_PROTOTYPE"
  --layer "$LAYER"
  --expected_fold 0
  --max_new_tokens 50
  --repeats 5
  --warmup_samples 2
  --warmup_tokens 8
  --preset null
  --spherical_kappa 20.0
  --spherical_alpha 0.7
  --spherical_beta -0.15
  --s2sl_strength 3.0
  --s2sl_slerp_t 0.2
  --s2sr_weight_kappa 3
  --s2sr_margin -0.7
  --output_dir "$OUTPUT_DIR"
)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
[[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max_samples "$MAX_SAMPLES")
[[ -n "${REPEATS:-}" ]] && ARGS+=(--repeats "$REPEATS")

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$HERE/benchmark.py" "${ARGS[@]}"
