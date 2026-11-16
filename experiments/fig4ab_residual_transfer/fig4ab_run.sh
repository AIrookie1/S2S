#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
FAMILY="${FAMILY:-LLAMA}"
METHOD="${METHOD:-S2S_R}"

case "$FAMILY" in
  LLAMA|QWEN) ;;
  *)
    echo "Error: FAMILY must be LLAMA or QWEN (got $FAMILY)." >&2
    exit 2
    ;;
esac

case "$METHOD" in
  S2S_R)
    source "$REPO_ROOT/configs/S2S_R/$FAMILY/truthfulqa.sh"
    CHECKPOINT_METHOD_DIR="S2S_R"
    METHOD_WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"
    METHOD_MARGIN="${MARGIN:--0.7}"
    # Parsed but unused by S2S_R; keeping explicit values makes evaluate.py's
    # output schema identical across both supported methods.
    METHOD_STRENGTH="${STRENGTH:-3.0}"
    METHOD_SLERP_T="${SLERP_T:-0.2}"
    ;;
  S2S_L_Static)
    source "$REPO_ROOT/configs/S2S_L/$FAMILY/truthfulqa.sh"
    CHECKPOINT_METHOD_DIR="S2S_L"
    METHOD_STRENGTH="${STRENGTH}"
    METHOD_SLERP_T="${SLERP_T}"
    # Parsed but unused by S2S_L_Static.
    METHOD_WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"
    METHOD_MARGIN="${MARGIN:--0.7}"
    ;;
  *)
    echo "Error: METHOD must be S2S_R or S2S_L_Static (got $METHOD)." >&2
    exit 2
    ;;
esac

source "$REPO_ROOT/scripts/common.sh"

# S2S-R keeps the released cross-fitted Spherical-Steering mean-only baseline.
# S2S-L-Static instead uses standard linear additive mean steering
# h <- h + strength * v_avg, so it has no spherical prototype dependency.
SPHERICAL_PROTOTYPE_PATHS=()
if [[ "$METHOD" == "S2S_R" ]]; then
  SPHERICAL_CONFIG="$REPO_ROOT/configs/spherical_steering/$FAMILY/truthfulqa.sh"
  mapfile -t _SPHERICAL_DEFAULTS < <(
    env -u MODEL -u MODEL_DIR -u MODEL_LABEL -u LAYER -u KAPPA -u ALPHA -u BETA -u NUM_FOLDS \
      bash -c 'source "$1"; printf "%s\n" "$MODEL" "$MODEL_DIR" "$LAYER" "$KAPPA" "$ALPHA" "$BETA" "$NUM_FOLDS"' \
      _ "$SPHERICAL_CONFIG"
  )
  SPHERICAL_MODEL="${SPHERICAL_MODEL:-${_SPHERICAL_DEFAULTS[0]}}"
  SPHERICAL_MODEL_DIR="${SPHERICAL_MODEL_DIR:-${_SPHERICAL_DEFAULTS[1]}}"
  SPHERICAL_LAYER="${SPHERICAL_LAYER:-${_SPHERICAL_DEFAULTS[2]}}"
  SPHERICAL_KAPPA="${SPHERICAL_KAPPA:-${_SPHERICAL_DEFAULTS[3]}}"
  SPHERICAL_ALPHA="${SPHERICAL_ALPHA:-${_SPHERICAL_DEFAULTS[4]}}"
  SPHERICAL_BETA="${SPHERICAL_BETA:-${_SPHERICAL_DEFAULTS[5]}}"
  SPHERICAL_NUM_FOLDS="${SPHERICAL_NUM_FOLDS:-${_SPHERICAL_DEFAULTS[6]}}"

  if [[ "$SPHERICAL_MODEL" != "$MODEL" ]]; then
    echo "Error: Spherical-Steering model ($SPHERICAL_MODEL) must match $METHOD model ($MODEL)." >&2
    exit 2
  fi
  if [[ "$SPHERICAL_MODEL_DIR" != "$MODEL_DIR" ]]; then
    echo "[residual-analysis] note: spherical MODEL_DIR differs from $METHOD MODEL_DIR;" >&2
    echo "the residual experiment uses the selected method's MODEL_DIR for all forward passes." >&2
  fi

  SPHERICAL_PROTO_DIR="${SPHERICAL_PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
  if [[ -n "${SPHERICAL_PROTOTYPE_PATHS_CSV:-}" ]]; then
    IFS=':' read -r -a SPHERICAL_PROTOTYPE_PATHS <<< "$SPHERICAL_PROTOTYPE_PATHS_CSV"
  else
    for ((fold=0; fold<SPHERICAL_NUM_FOLDS; fold++)); do
      SPHERICAL_PROTOTYPE_PATHS+=(
        "$SPHERICAL_PROTO_DIR/${SPHERICAL_MODEL}_layer${SPHERICAL_LAYER}_fold${fold}.npz"
      )
    done
  fi
fi

GPU="${GPU:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/$CHECKPOINT_METHOD_DIR/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/fig4ab_residual_transfer/$METHOD/$MODEL/fold${FOLD_IDX}}"
NUM_SHUFFLE_SEEDS="${NUM_SHUFFLE_SEEDS:-100}"
MC_SHUFFLE_SEEDS="${MC_SHUFFLE_SEEDS:-5}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
QUINTILES="${QUINTILES:-5}"
SKIP_MC="${SKIP_MC:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

if [[ "$PLOT_ONLY" != 1 ]]; then
  if [[ ! -s "$CHECKPOINT_PATH" ]]; then
    echo "[residual-analysis] missing $METHOD checkpoint: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ "$METHOD" == "S2S_R" ]]; then
    for prototype_path in "${SPHERICAL_PROTOTYPE_PATHS[@]}"; do
      if [[ ! -s "$prototype_path" ]]; then
        echo "[residual-analysis] missing Spherical-Steering prototype: $prototype_path" >&2
        echo "[hint] Generate all spherical folds first, e.g.:" >&2
        echo "       bash scripts/spherical_steering/$FAMILY/extract_features.sh" >&2
        echo "       bash scripts/spherical_steering/$FAMILY/compute_prototypes.sh" >&2
        exit 1
      fi
    done
  fi

  ARGS=(
    "$MODEL"
    --model_dir "$MODEL_DIR"
    --checkpoint_path "$CHECKPOINT_PATH"
    --layer "$LAYER"
    --expected_fold "$FOLD_IDX"
    --eval_split test
    --steering_method "$METHOD"
    --weight_kappa "$METHOD_WEIGHT_KAPPA"
    --margin "$METHOD_MARGIN"
    --strength "$METHOD_STRENGTH"
    --slerp_t "$METHOD_SLERP_T"
    --num_shuffle_seeds "$NUM_SHUFFLE_SEEDS"
    --mc_shuffle_seeds "$MC_SHUFFLE_SEEDS"
    --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    --quintiles "$QUINTILES"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --seed "$SEED"
    --preset "$EVAL_PRESET"
    --output_dir "$OUTPUT_DIR"
  )
  if [[ "$METHOD" == "S2S_R" ]]; then
    ARGS+=(
      --spherical_prototype_paths "${SPHERICAL_PROTOTYPE_PATHS[@]}"
      --spherical_layer "$SPHERICAL_LAYER"
      --spherical_kappa "$SPHERICAL_KAPPA"
      --spherical_alpha "$SPHERICAL_ALPHA"
      --spherical_beta "$SPHERICAL_BETA"
    )
  fi
  [[ "${FEW_SHOT:-0}" == 1 ]] && ARGS+=(--few_shot)
  [[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
  [[ "$SKIP_MC" == 1 ]] && ARGS+=(--skip_mc)
  [[ "${SAVE_MC_SCORES:-0}" == 1 ]] && ARGS+=(--save_mc_scores)
  [[ -n "${MAX_QUESTIONS:-}" ]] && ARGS+=(--max_questions "$MAX_QUESTIONS")

  cd "$REPO_ROOT"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$HERE/evaluate.py" "${ARGS[@]}"
fi

PLOT_ARGS=(--input_dir "$OUTPUT_DIR" --output_prefix "$OUTPUT_DIR/fig4ab_residual_transfer")
[[ "$SKIP_MC" == 1 ]] && PLOT_ARGS+=(--allow_missing_mc)

cd "$REPO_ROOT"
"$PYTHON_BIN" "$HERE/fig4ab_plot.py" "${PLOT_ARGS[@]}"
