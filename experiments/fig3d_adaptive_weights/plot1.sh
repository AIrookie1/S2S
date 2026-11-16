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
WEIGHT_KAPPAS="${WEIGHT_KAPPAS:-5}"
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

if [[ "$AUTO_PLOT" == "1" ]]; then
  run_cmd "$PYTHON_BIN" "$HERE/fig3d_plot.py" \
    --input-dir "$DATA_DIR" \
    --output-dir "$FIGURE_DIR" \
    --model-label "$MODEL_LABEL"
fi

echo "fig3d_adaptive_weights complete: $FIGURE_DIR"
