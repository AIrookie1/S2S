#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn3d_sga/experiments/margin_ablation.sh"
GPU="${GPU:-1}"
source "$REPO_ROOT/experiments/common.sh"
read -r -a MODEL_KEYS <<< "${MODELS:-LLAMA QWEN}"
read -r -a FOLD_INDICES <<< "${FOLDS:-0 1}"
EXPERIMENT_RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/margin_ablation}"
SUMMARY_DIR="$EXPERIMENT_RESULT_DIR/summaries"
mkdir -p "$SUMMARY_DIR"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"; prepare_feature "$FEATURE_PATH"
  for FOLD_IDX in "${FOLD_INDICES[@]}"; do
    (( FOLD_IDX >= 0 && FOLD_IDX < NUM_FOLDS )) || { echo "Invalid fold: $FOLD_IDX" >&2; exit 2; }
    CHECKPOINT_DIR="$EXPERIMENT_RESULT_DIR/checkpoints/$MODEL/fold${FOLD_IDX}"; DETAIL_DIR="$EXPERIMENT_RESULT_DIR/evaluations/$MODEL/fold${FOLD_IDX}"
    SUMMARY_JSON="$SUMMARY_DIR/${MODEL}_fold${FOLD_IDX}.json"; CHECKPOINT="$CHECKPOINT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_baseline.npz"
    mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"; rm -f "$SUMMARY_JSON"
    echo "=== train once | model=$MODEL fold=$FOLD_IDX ==="
    train_dyn3d "$FEATURE_PATH" "$CHECKPOINT" "$FOLD_IDX" "$VAL_FRACTION" "$SEED" "$ROUTER_TOP_K" "$LOAD_BALANCE_LOSS_WEIGHT"
    for RUN_MARGIN in "${MARGINS[@]}"; do
      TAG="$(safe_tag "$RUN_MARGIN")"; EVAL_JSON="$DETAIL_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_margin${TAG}_eval.json"
      echo "=== margin=$RUN_MARGIN | evaluate ==="
      eval_dyn3d "$CHECKPOINT" "$EVAL_JSON" "$WEIGHT_KAPPA" "$RUN_MARGIN" "$SEED"
      run_cmd "$PYTHON_BIN" "$HERE/collect.py" --margin "$RUN_MARGIN" --model "$MODEL" --model-label "$MODEL_LABEL" --layer "$LAYER" --fold "$FOLD_IDX" --num-folds "$NUM_FOLDS" --eval-json "$EVAL_JSON" --checkpoint "$CHECKPOINT" --output "$SUMMARY_JSON"
    done
  done
done
if [[ "${AUTO_PLOT:-1}" == 1 && "$DRY_RUN" != 1 ]]; then
  bash "$HERE/plot.sh"
fi
