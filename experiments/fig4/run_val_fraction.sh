#!/usr/bin/env bash
set -euo pipefail

# Run Fig. 4 for both models and both folds. Results are isolated by model/fold.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
GPU="${GPU:-1}"

FIG4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG4_DIR/../../script/common.sh"
source "$FIG4_DIR/../model_config.sh"

read -r -a MODEL_KEYS <<< "${MODELS:-llama qwen}"
read -r -a FOLD_INDICES <<< "${FOLDS:-0 1}"

VAL_FRACTIONS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
NUM_FOLDS=2
POSITIVE_MODE=mean

SVD_RANK="${SVD_RANK:-300}"
ROUTER_HIDDEN="${ROUTER_HIDDEN:-256}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SCALE_LOSS_WEIGHT="${SCALE_LOSS_WEIGHT:-0.03}"
COSINE_LOSS_WEIGHT="${COSINE_LOSS_WEIGHT:-0.003}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

WEIGHT_KAPPA="${WEIGHT_KAPPA:-10}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DTYPE="${DTYPE:-float16}"

SUMMARY_DIR="$FIG4_DIR/summaries"
mkdir -p "$SUMMARY_DIR"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"

  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
  if [[ ! -s "$FEATURE_PATH" ]]; then
    echo "Missing feature file for $MODEL: $FEATURE_PATH" >&2
    echo "Extract features for this model before running Fig. 4." >&2
    exit 1
  fi

  for FOLD_IDX in "${FOLD_INDICES[@]}"; do
    if (( FOLD_IDX < 0 || FOLD_IDX >= NUM_FOLDS )); then
      echo "Invalid fold: $FOLD_IDX (NUM_FOLDS=$NUM_FOLDS)" >&2
      exit 2
    fi

    CHECKPOINT_DIR="$FIG4_DIR/checkpoints/$MODEL/fold${FOLD_IDX}"
    DETAIL_DIR="$FIG4_DIR/results/$MODEL/fold${FOLD_IDX}"
    SUMMARY_JSON="$SUMMARY_DIR/${MODEL}_fold${FOLD_IDX}.json"
    mkdir -p "$CHECKPOINT_DIR" "$DETAIL_DIR"
    rm -f "$SUMMARY_JSON"

    echo
    echo "################################################################"
    echo "Fig. 4 | model=$MODEL | layer=$LAYER | fold=$FOLD_IDX"
    echo "################################################################"

    for VAL_FRACTION in "${VAL_FRACTIONS[@]}"; do
      FRACTION_TAG="${VAL_FRACTION/./p}"
      CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}.npz"
      EVAL_JSON="$DETAIL_DIR/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}_eval.json"

      echo
      echo "========== val_fraction=$VAL_FRACTION | train =========="
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py \
        --feature_path "$FEATURE_PATH" \
        --positive_mode "$POSITIVE_MODE" \
        --output_path "$CHECKPOINT_PATH" \
        --num_folds "$NUM_FOLDS" \
        --fold_idx "$FOLD_IDX" \
        --val_fraction "$VAL_FRACTION" \
        --rank "$SVD_RANK" \
        --router_hidden "$ROUTER_HIDDEN" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --scale_loss_weight "$SCALE_LOSS_WEIGHT" \
        --cosine_loss_weight "$COSINE_LOSS_WEIGHT" \
        --use_data_mode "$USE_DATA_MODE" \
        --aux_interp_count "$AUX_INTERP_COUNT"
      test -s "$CHECKPOINT_PATH"

      echo "========== val_fraction=$VAL_FRACTION | evaluate =========="
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py \
        --method-dir dyn3d_sga \
        --entry-file evaluate_3d_dyn.py \
        -- \
        "$MODEL" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --model_dir "$MODEL_DIR" \
        --layer "$LAYER" \
        --eval_split "$EVAL_SPLIT" \
        --weight_kappa "$WEIGHT_KAPPA" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        --output_path "$EVAL_JSON" \
        --save_details
      test -s "$EVAL_JSON"

      "$PYTHON_BIN" "$FIG4_DIR/collect_mc1.py" \
        --val-fraction "$VAL_FRACTION" \
        --model "$MODEL" \
        --model-label "$MODEL_LABEL" \
        --layer "$LAYER" \
        --fold "$FOLD_IDX" \
        --num-folds "$NUM_FOLDS" \
        --eval-json "$EVAL_JSON" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$SUMMARY_JSON"
    done

    echo "Saved summary: $SUMMARY_JSON"
  done
done

echo
echo "Fig. 4 complete. Per-fold summaries: $SUMMARY_DIR"

if [[ "${AUTO_PLOT:-1}" == "1" ]]; then
  bash "$FIG4_DIR/plot.sh"
fi
