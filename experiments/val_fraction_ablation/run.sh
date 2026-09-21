#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn3d_sga/experiments/val_fraction_ablation.sh"
GPU="${GPU:-0}"
source "$REPO_ROOT/experiments/common.sh"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
read -r -a MODEL_KEYS <<< "${MODELS:-LLAMA QWEN}"
(( ${#SEED_VALUES[@]} >= 2 )) || { echo "At least two seeds are required." >&2; exit 2; }
[[ "$(printf '%s\n' "${SEED_VALUES[@]}" | sort -u | wc -l)" -eq "${#SEED_VALUES[@]}" ]] || { echo "SEEDS must be distinct." >&2; exit 2; }
EXPERIMENT_RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/val_fraction_ablation}"
SUMMARY_DIR="$EXPERIMENT_RESULT_DIR/summaries"; mkdir -p "$SUMMARY_DIR"
for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"
  (( FOLD_IDX >= 0 && FOLD_IDX < NUM_FOLDS )) || { echo "Invalid fold: $FOLD_IDX" >&2; exit 2; }
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"; prepare_feature "$FEATURE_PATH"
  SUMMARY_JSON="$SUMMARY_DIR/${MODEL}_fold${FOLD_IDX}.json"; rm -f "$SUMMARY_JSON"
  for RUN_SEED in "${SEED_VALUES[@]}"; do
    for FRACTION in "${VAL_FRACTIONS[@]}"; do
      TAG="$(safe_tag "$FRACTION")"
      CHECKPOINT="$EXPERIMENT_RESULT_DIR/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_seed${RUN_SEED}_val${TAG}.npz"
      EVAL_JSON="$EXPERIMENT_RESULT_DIR/evaluations/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_seed${RUN_SEED}_val${TAG}_eval.json"
      mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$EVAL_JSON")"
      if [[ "$DRY_RUN" != 1 && "$REUSE_EXISTING" == 1 && -s "$CHECKPOINT" && -s "$EVAL_JSON" ]]; then
        echo "=== val_fraction=$FRACTION seed=$RUN_SEED | reuse ==="
      else
        echo "=== val_fraction=$FRACTION seed=$RUN_SEED | train/eval ==="
        train_dyn3d "$FEATURE_PATH" "$CHECKPOINT" "$FOLD_IDX" "$FRACTION" "$RUN_SEED"
        eval_dyn3d "$CHECKPOINT" "$EVAL_JSON" "$WEIGHT_KAPPA" "$MARGIN" "$RUN_SEED"
      fi
      run_cmd "$PYTHON_BIN" "$HERE/collect.py" --val-fraction "$FRACTION" --seed "$RUN_SEED" --model "$MODEL" --model-label "$MODEL_LABEL" --layer "$LAYER" --fold "$FOLD_IDX" --num-folds "$NUM_FOLDS" --eval-json "$EVAL_JSON" --checkpoint "$CHECKPOINT" --output "$SUMMARY_JSON"
    done
  done
done
if [[ "${AUTO_PLOT:-1}" == 1 && "$DRY_RUN" != 1 ]]; then
  SEEDS="${SEED_VALUES[*]}" bash "$HERE/plot.sh"
fi
