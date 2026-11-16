#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/configs/S2S_L/experiments/fig3a_s2s_l_hyperparameters.sh"
source "$REPO_ROOT/scripts/common.sh"

GPU="${GPU:-2}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"
RUN_TRAIN="${RUN_TRAIN:-auto}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"
AUTO_PLOT="${AUTO_PLOT:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/fig3a_s2s_l_hyperparameters}"
read -r -a MODEL_KEYS <<< "${MODELS:-LLAMA QWEN}"
read -r -a FOLD_INDICES <<< "${FOLDS:-0 1}"
mkdir -p "$OUTPUT_DIR"

case "$RUN_EXTRACT" in auto|0|1) ;; *) echo "RUN_EXTRACT must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$RUN_TRAIN" in auto|0|1) ;; *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$REUSE_EXISTING" in 0|1) ;; *) echo "REUSE_EXISTING must be 0 or 1" >&2; exit 2 ;; esac

configure_S2S_L_model() {
  local key="${1^^}"
  unset MODEL MODEL_DIR MODEL_LABEL LAYER FOLD_IDX DEFAULT_LAYER DEFAULT_FOLD_IDX NUM_HIDDEN_LAYERS
  case "$key" in
    LLAMA|QWEN) source "$REPO_ROOT/configs/S2S_L/$key/truthfulqa.sh" ;;
    *) echo "Unknown model: $1 (expected LLAMA or QWEN)" >&2; return 2 ;;
  esac
}

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_S2S_L_model "$MODEL_KEY"
  MODEL_GPU_VAR="${MODEL_KEY^^}_GPU"
  MODEL_GPU="${!MODEL_GPU_VAR:-$GPU}"
  FEATURE_OVERRIDE_VAR="${MODEL_KEY^^}_FEATURE_PATH"
  FEATURE_PATH="${!FEATURE_OVERRIDE_VAR:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"

  if [[ "$RUN_EXTRACT" == 1 || ( "$RUN_EXTRACT" == auto && ! -s "$FEATURE_PATH" ) ]]; then
    mkdir -p "$(dirname "$FEATURE_PATH")"
    CUDA_VISIBLE_DEVICES="$MODEL_GPU" "$PYTHON_BIN" methods/S2S_L/extract_features.py "$MODEL" \
      --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$(dirname "$FEATURE_PATH")" \
      --output_name "$(basename "$FEATURE_PATH")" --batch_size "$EXTRACT_BATCH_SIZE" \
      --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
  fi
  [[ -s "$FEATURE_PATH" ]] || { echo "Missing feature file: $FEATURE_PATH" >&2; exit 1; }

  FOLD_CSVS=()
  for FOLD_IDX in "${FOLD_INDICES[@]}"; do
    (( FOLD_IDX >= 0 && FOLD_IDX < NUM_FOLDS )) || { echo "Invalid fold: $FOLD_IDX" >&2; exit 2; }
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_predictor_${POSITIVE_MODE}.npz"
    FOLD_OUTPUT_DIR="$OUTPUT_DIR/raw/$MODEL/fold${FOLD_IDX}"
    mkdir -p "$(dirname "$CHECKPOINT_PATH")" "$FOLD_OUTPUT_DIR"

    if [[ "$RUN_TRAIN" == 1 || ( "$RUN_TRAIN" == auto && ! -s "$CHECKPOINT_PATH" ) ]]; then
      echo "=== S2S_L train | model=$MODEL fold=$FOLD_IDX ==="
      CUDA_VISIBLE_DEVICES="$MODEL_GPU" "$PYTHON_BIN" methods/S2S_L/train_predictor.py \
        --feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" \
        --output_path "$CHECKPOINT_PATH" --num_folds "$NUM_FOLDS" --fold_idx "$FOLD_IDX" \
        --val_fraction "$VAL_FRACTION" --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" \
        --device "$DEVICE" --seed "$SEED"
    fi
    [[ -s "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }

    echo "=== S2S_L grid | model=$MODEL fold=$FOLD_IDX ==="
    EVAL_ARGS=(
      "$MODEL"
      --model_dir "$MODEL_DIR"
      --checkpoint_path "$CHECKPOINT_PATH"
      --expected_fold "$FOLD_IDX"
      --layer "$LAYER"
      --eval_split "$EVAL_SPLIT"
      --strengths "${S2S_L_ABLATION_STRENGTHS[*]}"
      --slerps "${S2S_L_ABLATION_SLERPS[*]}"
      --device "$DEVICE"
      --dtype "$DTYPE"
      --seed "$SEED"
      --preset "$EVAL_PRESET"
      --output_dir "$FOLD_OUTPUT_DIR"
    )
    if [[ "$REUSE_EXISTING" == 1 ]]; then EVAL_ARGS+=(--resume); fi
    # EXTRA_ARGS is intentionally word-split so callers can pass additional CLI flags.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS:-})
    CUDA_VISIBLE_DEVICES="$MODEL_GPU" "$PYTHON_BIN" "$HERE/evaluate_grid.py" "${EVAL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}"
    FOLD_CSVS+=("$FOLD_OUTPUT_DIR/S2S_L_ablation_mc1.csv")
  done

  SUMMARY_DIR="$OUTPUT_DIR/summaries"
  mkdir -p "$SUMMARY_DIR"
  AGG_CSV="$SUMMARY_DIR/${MODEL}_fold_mean.csv"
  AGG_JSON="$SUMMARY_DIR/${MODEL}_fold_mean.json"
  "$PYTHON_BIN" "$HERE/aggregate.py" \
    --model "$MODEL" --model-label "$MODEL_LABEL" \
    --fold-csvs "${FOLD_CSVS[@]}" \
    --expected-folds "${FOLD_INDICES[*]}" \
    --output-csv "$AGG_CSV" --output-json "$AGG_JSON"
done

if [[ "$AUTO_PLOT" == 1 ]]; then
  OUTPUT_DIR="$OUTPUT_DIR" bash "$HERE/fig3a_plot.sh"
fi
