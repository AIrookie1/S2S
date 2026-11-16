#!/usr/bin/env bash
set -euo pipefail

# Token-level ternary density of S2S_R adaptive (alpha, beta, gamma) weights.
# Defaults: Llama, fold0, kappa={1,3,5,10}, current S2S_R checkpoint layout.
#
# Useful overrides:
#   WEIGHTS_GATE_MODE=all|steered   # include all tokens or only margin-gated tokens
#   RUN_TRAIN=1                     # train the standard S2S_R checkpoint if missing
#   RUN_EXTRACT=1                   # additionally extract features if training needs them
#   SKIP_EVAL=1                     # reuse existing weight JSON files
#   CHECKPOINT=/path/to/file.npz    # explicit checkpoint override

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

GPU="${GPU:-2}"
MODEL_KEY="${MODEL_KEY:-LLAMA}"
FOLD_IDX="${FOLD_IDX:-0}"
WEIGHT_KAPPAS="${WEIGHT_KAPPAS:-20}"
WEIGHTS_BINS="${WEIGHTS_BINS:-80}"
WEIGHTS_GATE_MODE="${WEIGHTS_GATE_MODE:-all}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
AUTO_PLOT="${AUTO_PLOT:-1}"

source "$REPO_ROOT/experiments/common.sh"
configure_model "$MODEL_KEY"

case "$WEIGHTS_GATE_MODE" in
  all|steered) ;;
  *) echo "WEIGHTS_GATE_MODE must be 'all' or 'steered'" >&2; exit 2 ;;
esac
case "$RUN_TRAIN" in 0|1) ;; *) echo "RUN_TRAIN must be 0 or 1" >&2; exit 2 ;; esac
case "$SKIP_EVAL" in 0|1) ;; *) echo "SKIP_EVAL must be 0 or 1" >&2; exit 2 ;; esac

(( FOLD_IDX >= 0 && FOLD_IDX < NUM_FOLDS )) || {
  echo "Invalid fold: $FOLD_IDX (NUM_FOLDS=$NUM_FOLDS)" >&2
  exit 2
}

FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
EXPERIMENT_RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$RESULT_ROOT/experiment/fig3d_adaptive_weights}"
DATA_DIR="${DATA_DIR:-$EXPERIMENT_RESULT_DIR/data/$WEIGHTS_GATE_MODE}"
FIGURE_DIR="${FIGURE_DIR:-$EXPERIMENT_RESULT_DIR/figures/$WEIGHTS_GATE_MODE}"
mkdir -p "$DATA_DIR" "$FIGURE_DIR"

if [[ ! -s "$CHECKPOINT" && "$DRY_RUN" != "1" ]]; then
  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo "========== checkpoint missing | train current S2S_R =========="
    prepare_feature "$FEATURE_PATH"
    mkdir -p "$(dirname "$CHECKPOINT")"
    train_S2S_R "$FEATURE_PATH" "$CHECKPOINT" "$FOLD_IDX" "$VAL_FRACTION" "$SEED"
  else
    echo "Missing current S2S_R checkpoint: $CHECKPOINT" >&2
    echo "Set RUN_TRAIN=1 to train it, or set CHECKPOINT=/path/to/checkpoint.npz." >&2
    exit 1
  fi
fi

read -r -a KAPPAS <<< "$WEIGHT_KAPPAS"
for RUN_KAPPA in "${KAPPAS[@]}"; do
  TAG="$(safe_tag "$RUN_KAPPA")"
  EVAL_JSON="$DATA_DIR/${MODEL}_fold${FOLD_IDX}_kappa${TAG}_eval.json"
  WEIGHTS_JSON="$DATA_DIR/${MODEL}_fold${FOLD_IDX}_kappa${TAG}_weights.json"

  if [[ "$SKIP_EVAL" == "1" && -s "$WEIGHTS_JSON" ]]; then
    echo "========== kappa=$RUN_KAPPA | reuse $WEIGHTS_GATE_MODE density data =========="
    continue
  fi

  echo "========== kappa=$RUN_KAPPA | gate=$WEIGHTS_GATE_MODE | evaluate =========="
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py \
    --method-dir S2S_R \
    --entry-file evaluate_S2S_R.py \
    -- \
    "$MODEL" \
    --checkpoint_path "$CHECKPOINT" \
    --model_dir "$MODEL_DIR" \
    --layer "$LAYER" \
    --eval_split "$EVAL_SPLIT" \
    --weight_kappa "$RUN_KAPPA" \
    --margin "$MARGIN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --output_path "$EVAL_JSON" \
    --weights_output_path "$WEIGHTS_JSON" \
    --weights_bins "$WEIGHTS_BINS" \
    --weights_gate_mode "$WEIGHTS_GATE_MODE" \
    --save_details

  require_file "$EVAL_JSON"
  require_file "$WEIGHTS_JSON"
done

if [[ "$AUTO_PLOT" == "1" ]]; then
  run_cmd "$PYTHON_BIN" "$HERE/fig3d_plot.py" \
    --input-dir "$DATA_DIR" \
    --output-dir "$FIGURE_DIR" \
    --model-label "$MODEL_LABEL"
fi

echo "fig3d_adaptive_weights complete: $FIGURE_DIR"
