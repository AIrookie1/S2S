#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"
DRY_RUN="${DRY_RUN:-0}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
GPU=2
[[ "$RUN_EXTRACT" =~ ^(auto|0|1)$ && "$RUN_TRAIN" =~ ^[01]$ && "$RUN_EVAL" =~ ^[01]$ ]] || {
  echo "RUN_EXTRACT must be auto/0/1; RUN_TRAIN and RUN_EVAL must be 0/1" >&2; exit 2;
}
read -r -a DATASET_KEYS <<< "${DATASETS:-boolq copa storycloze mmlu_global winogrande}"
EXPERIMENT_RESULT_DIR="${EXPERIMENT_RESULT_DIR:-$RESULT_ROOT/experiment/rebuttal_R2Q2_cross_task_transfer}"
run_cmd(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
require_file(){ [[ "$DRY_RUN" == 1 || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }
# Each dataset gets a fresh shell so sourced defaults cannot leak to the next one.
for DATASET_KEY in "${DATASET_KEYS[@]}"; do (
  case "$DATASET_KEY" in boolq|copa|storycloze|mmlu_global|winogrande) ;; *) echo "Unknown dataset: $DATASET_KEY" >&2; exit 2 ;; esac
  source "$REPO_ROOT/configs/S2S_R/LLAMA/generic/$DATASET_KEY.sh"
  TRUTHFULQA_FEATURE="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
  DATASET_DIR="$EXPERIMENT_RESULT_DIR/$MODEL/$DATASET"
  if [[ "$RUN_EXTRACT" == 1 || ( "$RUN_EXTRACT" == auto && ! -s "$TRUTHFULQA_FEATURE" ) ]]; then
    run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/S2S_R/extract_features.py "$MODEL" \
      --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$FEATURE_DIR" \
      --output_name "$(basename "$TRUTHFULQA_FEATURE")" --batch_size "$EXTRACT_BATCH_SIZE" \
      --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
  fi
  if [[ "$RUN_TRAIN" == 1 || "$RUN_EVAL" == 1 ]]; then
    require_file "$TRUTHFULQA_FEATURE"
    # Reject a source feature file with a wrong model/layer or fixed fold split.
    run_cmd "$PYTHON_BIN" "$HERE/prepare_checkpoint.py" --validate-source \
      --truthfulqa-feature "$TRUTHFULQA_FEATURE" --model "$MODEL" --layer "$LAYER"
  fi
  EVALUATIONS=()
  for SOURCE_FOLD in 0 1; do
    SOURCE_CHECKPOINT="$DATASET_DIR/checkpoints/truthfulqa_fold${SOURCE_FOLD}.npz"
    TRANSFER_CHECKPOINT="$DATASET_DIR/checkpoints/transfer_fold${SOURCE_FOLD}.npz"
    EVAL_JSON="$DATASET_DIR/evaluations/truthfulqa_mean_fold${SOURCE_FOLD}.json"
    if [[ "$RUN_TRAIN" == 1 ]]; then
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/S2S_R/train_predictor.py \
        --feature_path "$TRUTHFULQA_FEATURE" --output_path "$SOURCE_CHECKPOINT" \
        --positive_mode "$POSITIVE_MODE" --num_folds 2 --fold_idx "$SOURCE_FOLD" \
        --val_fraction "$VAL_FRACTION" --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$SEED"
    fi
    if [[ "$RUN_TRAIN" == 1 || "$RUN_EVAL" == 1 ]]; then
      require_file "$SOURCE_CHECKPOINT"
      run_cmd "$PYTHON_BIN" "$HERE/prepare_checkpoint.py" \
        --source-checkpoint "$SOURCE_CHECKPOINT" --truthfulqa-feature "$TRUTHFULQA_FEATURE" \
        --dataset "$DATASET" --model "$MODEL" --layer "$LAYER" \
        --source-fold "$SOURCE_FOLD" --output "$TRANSFER_CHECKPOINT"
    fi
    if [[ "$RUN_EVAL" == 1 ]]; then
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/S2S_R/generic/evaluate_generic.py \
        --model_name "$MODEL" --model_dir "$MODEL_DIR" --dataset "$DATASET" --layer "$LAYER" \
        --checkpoint_path "$TRANSFER_CHECKPOINT" --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN" \
        --method S2S_R --seed "$SEED" --dtype "$DTYPE" --output_path "$EVAL_JSON"
      EVALUATIONS+=("$EVAL_JSON")
    fi
  done
  if [[ "$RUN_EVAL" == 1 ]]; then
    run_cmd "$PYTHON_BIN" "$HERE/collect.py" --evaluations "${EVALUATIONS[@]}" --output-dir "$DATASET_DIR"
  fi
); done
if [[ "$RUN_EVAL" == 1 ]]; then
  SUMMARIES=()
  for DATASET_KEY in "${DATASET_KEYS[@]}"; do
    SUMMARIES+=("$EXPERIMENT_RESULT_DIR/${LLAMA_MODEL:-llama3.1-8B-Instruct}/$DATASET_KEY/summary.json")
  done
  run_cmd "$PYTHON_BIN" "$HERE/collect.py" --summaries "${SUMMARIES[@]}" --output-dir "$EXPERIMENT_RESULT_DIR"
fi
