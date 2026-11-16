#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/configs/S2S_R/experiments/fig3b_hidden_size.sh"
GPU="${GPU:-2}"
BASE_GPU="$GPU"
source "$REPO_ROOT/experiments/common.sh"

REUSE_EXISTING="${REUSE_EXISTING:-1}"
read -r -a MODEL_KEYS <<< "${MODELS:-LLAMA QWEN}"
read -r -a FOLD_INDICES <<< "${FOLDS:-0 1}"
EXPERIMENT_RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$REPO_ROOT/results/experiment/fig3b_hidden_size}"
SUMMARY_DIR="$EXPERIMENT_RESULT_DIR/summaries"
mkdir -p "$SUMMARY_DIR"
case "$REUSE_EXISTING" in 0|1) ;; *) echo "REUSE_EXISTING must be 0 or 1" >&2; exit 2 ;; esac

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"
  MODEL_GPU_VAR="${MODEL_KEY^^}_GPU"
  GPU="${!MODEL_GPU_VAR:-$BASE_GPU}"
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
  prepare_feature "$FEATURE_PATH"

  for FOLD_IDX in "${FOLD_INDICES[@]}"; do
    (( FOLD_IDX >= 0 && FOLD_IDX < NUM_FOLDS )) || { echo "Invalid fold: $FOLD_IDX" >&2; exit 2; }
    SUMMARY_JSON="$SUMMARY_DIR/${MODEL}_fold${FOLD_IDX}.json"
    # Rebuild the compact summary from exactly the requested hidden-size grid.
    rm -f "$SUMMARY_JSON"

    for HIDDEN in "${MLP_HIDDEN_VALUES[@]}"; do
      CHECKPOINT="$EXPERIMENT_RESULT_DIR/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_hidden${HIDDEN}.npz"
      EVAL_JSON="$EXPERIMENT_RESULT_DIR/evaluations/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_hidden${HIDDEN}_eval.json"
      mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$EVAL_JSON")"

      CHECKPOINT_VALID=0
      if [[ "$REUSE_EXISTING" == 1 && -s "$CHECKPOINT" ]]; then
        if "$PYTHON_BIN" "$HERE/validate_checkpoint.py" \
          --checkpoint "$CHECKPOINT" --mlp-hidden "$HIDDEN" \
          --fold "$FOLD_IDX" --num-folds "$NUM_FOLDS" \
          --val-fraction "$VAL_FRACTION" --positive-mode "$POSITIVE_MODE" \
          --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
          --learning-rate "$LEARNING_RATE" --seed "$SEED"; then
          CHECKPOINT_VALID=1
        else
          echo "=== cached checkpoint does not match this ablation cell; retraining ==="
        fi
      fi

      CHECKPOINT_REBUILT=0
      if [[ "$REUSE_EXISTING" != 1 || "$CHECKPOINT_VALID" != 1 ]]; then
        echo "=== S2S_R MLP hidden=$HIDDEN | model=$MODEL fold=$FOLD_IDX | train ==="
        # Call the S2S_R trainer directly so this experiment never depends on
        # an older common.sh helper that may ignore the hidden-size argument.
        run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
          methods/S2S_R/train_predictor.py \
          --feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" \
          --output_path "$CHECKPOINT" --num_folds "$NUM_FOLDS" \
          --fold_idx "$FOLD_IDX" --val_fraction "$VAL_FRACTION" \
          --mlp_hidden "$HIDDEN" --epochs "$EPOCHS" \
          --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" \
          --device "$DEVICE" --seed "$SEED"
        require_file "$CHECKPOINT"
        if [[ "$DRY_RUN" != 1 ]]; then
          "$PYTHON_BIN" "$HERE/validate_checkpoint.py" \
            --checkpoint "$CHECKPOINT" --mlp-hidden "$HIDDEN" \
            --fold "$FOLD_IDX" --num-folds "$NUM_FOLDS" \
            --val-fraction "$VAL_FRACTION" --positive-mode "$POSITIVE_MODE" \
            --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --learning-rate "$LEARNING_RATE" --seed "$SEED"
        fi
        CHECKPOINT_REBUILT=1
      else
        echo "=== S2S_R MLP hidden=$HIDDEN | model=$MODEL fold=$FOLD_IDX | reuse checkpoint ==="
      fi

      EVAL_VALID=0
      if [[ "$REUSE_EXISTING" == 1 && "$CHECKPOINT_REBUILT" != 1 && -s "$EVAL_JSON" ]]; then
        # An evaluation older than its checkpoint is stale even when the path
        # is unchanged, because the checkpoint may have been retrained in place.
        if [[ ! "$CHECKPOINT" -nt "$EVAL_JSON" ]] && \
          "$PYTHON_BIN" "$HERE/validate_evaluation.py" \
            --evaluation "$EVAL_JSON" --model "$MODEL" --layer "$LAYER" \
            --eval-split "$EVAL_SPLIT" --checkpoint "$CHECKPOINT" \
            --weight-kappa "$WEIGHT_KAPPA" --margin "$MARGIN"; then
          EVAL_VALID=1
        else
          echo "=== cached evaluation is stale or mismatched; reevaluating ==="
        fi
      fi

      if [[ "$REUSE_EXISTING" != 1 || "$EVAL_VALID" != 1 ]]; then
        echo "=== S2S_R MLP hidden=$HIDDEN | model=$MODEL fold=$FOLD_IDX | evaluate ==="
        eval_S2S_R "$CHECKPOINT" "$EVAL_JSON" "$WEIGHT_KAPPA" "$MARGIN" "$SEED"
        if [[ "$DRY_RUN" != 1 ]]; then
          "$PYTHON_BIN" "$HERE/validate_evaluation.py" \
            --evaluation "$EVAL_JSON" --model "$MODEL" --layer "$LAYER" \
            --eval-split "$EVAL_SPLIT" --checkpoint "$CHECKPOINT" \
            --weight-kappa "$WEIGHT_KAPPA" --margin "$MARGIN"
        fi
      else
        echo "=== S2S_R MLP hidden=$HIDDEN | model=$MODEL fold=$FOLD_IDX | reuse evaluation ==="
      fi

      run_cmd "$PYTHON_BIN" "$HERE/collect.py" \
        --mlp-hidden "$HIDDEN" --model "$MODEL" --model-label "$MODEL_LABEL" \
        --layer "$LAYER" --fold "$FOLD_IDX" --num-folds "$NUM_FOLDS" \
        --eval-json "$EVAL_JSON" --checkpoint "$CHECKPOINT" --output "$SUMMARY_JSON"
    done
  done
done

if [[ "$DRY_RUN" != 1 ]]; then
  "$PYTHON_BIN" "$HERE/aggregate.py" \
    --input-dir "$SUMMARY_DIR" --num-folds "$NUM_FOLDS" \
    --output-csv "$EXPERIMENT_RESULT_DIR/mlp_hidden_ablation_metrics.csv" \
    --output-json "$EXPERIMENT_RESULT_DIR/mlp_hidden_ablation_metrics.json"
fi

if [[ "${AUTO_PLOT:-1}" == 1 && "$DRY_RUN" != 1 ]]; then
  bash "$HERE/fig3b_hidden_plot.sh"
fi
