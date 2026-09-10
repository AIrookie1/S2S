#!/usr/bin/env bash
set -euo pipefail

# Table 1: ablate the vector MSE, scale MSE and cosine-direction losses.
# Configurations: mse_only, mse_scale, cos_only, cos_scale, all.

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
GPU="${GPU:-2}"
TABLE1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TABLE1_DIR/../../script/common.sh"
source "$TABLE1_DIR/../model_config.sh"

read -r -a MODEL_KEYS <<< "${MODELS:-llama qwen}"
read -r -a FOLD_INDICES <<< "${FOLDS:-0}"
read -r -a CONFIG_KEYS <<< "${CONFIGS:-mse_only mse_scale cos_only cos_scale all}"

NUM_FOLDS="${NUM_FOLDS:-2}"
VAL_FRACTION="${VAL_FRACTION:-0.5}"
POSITIVE_MODE="${POSITIVE_MODE:-mean}"
SVD_RANK="${SVD_RANK:-300}"
ROUTER_HIDDEN="${ROUTER_HIDDEN:-256}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
WEIGHT_KAPPA="${WEIGHT_KAPPA:-10}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DTYPE="${DTYPE:-float16}"

MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
SCALE_WEIGHT="${SCALE_WEIGHT:-0.03}"
COSINE_WEIGHT="${COSINE_WEIGHT:-0.003}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
REUSE_FIG4_ALL="${REUSE_FIG4_ALL:-1}"
SUMMARY_JSON="${SUMMARY_JSON:-$TABLE1_DIR/table1_results.json}"

if [[ "${RESET_SUMMARY:-1}" == "1" ]]; then
  rm -f "$SUMMARY_JSON"
elif [[ "${RESET_SUMMARY:-1}" != "0" ]]; then
  echo "RESET_SUMMARY must be 0 or 1" >&2
  exit 2
fi

loss_args() {
  case "$1" in
    mse_only)
      VECTOR_LOSS_WEIGHT="$MSE_WEIGHT"; SCALE_LOSS_WEIGHT=0; COSINE_LOSS_WEIGHT=0 ;;
    mse_scale)
      VECTOR_LOSS_WEIGHT="$MSE_WEIGHT"; SCALE_LOSS_WEIGHT="$SCALE_WEIGHT"; COSINE_LOSS_WEIGHT=0 ;;
    cos_only)
      VECTOR_LOSS_WEIGHT=0; SCALE_LOSS_WEIGHT=0; COSINE_LOSS_WEIGHT="$COSINE_WEIGHT" ;;
    cos_scale)
      VECTOR_LOSS_WEIGHT=0; SCALE_LOSS_WEIGHT="$SCALE_WEIGHT"; COSINE_LOSS_WEIGHT="$COSINE_WEIGHT" ;;
    all)
      VECTOR_LOSS_WEIGHT="$MSE_WEIGHT"; SCALE_LOSS_WEIGHT="$SCALE_WEIGHT"; COSINE_LOSS_WEIGHT="$COSINE_WEIGHT" ;;
    *)
      echo "Unknown loss configuration: $1" >&2
      echo "Expected mse_only, mse_scale, cos_only, cos_scale or all." >&2
      return 2 ;;
  esac
}

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
  if [[ ! -s "$FEATURE_PATH" ]]; then
    echo "Missing feature file for $MODEL: $FEATURE_PATH" >&2
    exit 1
  fi

  for FOLD_IDX in "${FOLD_INDICES[@]}"; do
    if (( FOLD_IDX < 0 || FOLD_IDX >= NUM_FOLDS )); then
      echo "Invalid fold=$FOLD_IDX for NUM_FOLDS=$NUM_FOLDS" >&2
      exit 2
    fi
    for CONFIG in "${CONFIG_KEYS[@]}"; do
      loss_args "$CONFIG"
      FRACTION_TAG="${VAL_FRACTION/./p}"
      CHECKPOINT_PATH="$TABLE1_DIR/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}_seed${SEED}_${CONFIG}.npz"
      EVAL_JSON="$TABLE1_DIR/results/$MODEL/fold${FOLD_IDX}/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}_seed${SEED}_${CONFIG}_eval.json"
      mkdir -p "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$EVAL_JSON")"

      SOURCE_CHECKPOINT="$TABLE1_DIR/../fig4/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}.npz"
      SOURCE_EVAL="$TABLE1_DIR/../fig4/results/$MODEL/fold${FOLD_IDX}/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_val${FRACTION_TAG}_eval.json"
      if [[ "$CONFIG" == "all" && "$REUSE_FIG4_ALL" == "1" && "$FOLD_IDX" == "0" && "$SEED" == "42" && -s "$SOURCE_CHECKPOINT" && -s "$SOURCE_EVAL" ]]; then
        CHECKPOINT_PATH="$SOURCE_CHECKPOINT"
        EVAL_JSON="$SOURCE_EVAL"
        echo "========== model=$MODEL fold=$FOLD_IDX config=$CONFIG | reuse Fig. 4 =========="
      elif [[ "$REUSE_EXISTING" == "1" && -s "$CHECKPOINT_PATH" && -s "$EVAL_JSON" ]]; then
        echo "========== model=$MODEL fold=$FOLD_IDX config=$CONFIG | reuse Table 1 artifacts =========="
      else
        echo "========== model=$MODEL fold=$FOLD_IDX config=$CONFIG | train =========="
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
          --vector_loss_weight "$VECTOR_LOSS_WEIGHT" \
          --scale_loss_weight "$SCALE_LOSS_WEIGHT" \
          --cosine_loss_weight "$COSINE_LOSS_WEIGHT" \
          --use_data_mode "$USE_DATA_MODE" \
          --aux_interp_count "$AUX_INTERP_COUNT"
        test -s "$CHECKPOINT_PATH"

        echo "========== model=$MODEL fold=$FOLD_IDX config=$CONFIG | evaluate =========="
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
      fi

      "$PYTHON_BIN" "$TABLE1_DIR/collect_table1.py" \
        --output "$SUMMARY_JSON" \
        --model "$MODEL" \
        --model-label "$MODEL_LABEL" \
        --layer "$LAYER" \
        --fold "$FOLD_IDX" \
        --num-folds "$NUM_FOLDS" \
        --config "$CONFIG" \
        --vector-loss-weight "$VECTOR_LOSS_WEIGHT" \
        --scale-loss-weight "$SCALE_LOSS_WEIGHT" \
        --cosine-loss-weight "$COSINE_LOSS_WEIGHT" \
        --val-fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        --eval-json "$EVAL_JSON" \
        --checkpoint "$CHECKPOINT_PATH"
    done
  done
done

echo "Table 1 complete: $SUMMARY_JSON"
