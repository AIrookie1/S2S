#!/usr/bin/env bash
set -euo pipefail

# Fig. 4a: use fold0 only and repeat each validation fraction with two seeds.
# The default seed 42 artifacts from Fig. 4 can be reused by setting
# REUSE_FIG4_SEED42=1 (the default); a second seed is trained independently.

USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
GPU="${GPU:-1}"

FIG4A_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG4A_DIR/../../script/common.sh"
source "$FIG4A_DIR/../model_config.sh"

read -r -a MODEL_KEYS <<< "${MODELS:-llama qwen}"
read -r -a SEED_VALUES <<< "${SEEDS:-42 123 0}"
if (( ${#SEED_VALUES[@]} < 2 )); then
  echo "Fig. 4a requires at least two distinct seeds (SEEDS='42 123 0')." >&2
  exit 2
fi
for ((i = 0; i < ${#SEED_VALUES[@]}; i++)); do
  for ((j = i + 1; j < ${#SEED_VALUES[@]}; j++)); do
    if [[ "${SEED_VALUES[$i]}" == "${SEED_VALUES[$j]}" ]]; then
      echo "SEEDS must contain distinct values: ${SEED_VALUES[*]}" >&2
      exit 2
    fi
  done
done

VAL_FRACTIONS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FOLD_IDX=0
NUM_FOLDS=2
POSITIVE_MODE=mean

SVD_RANK="${SVD_RANK:-300}"
ROUTER_HIDDEN="${ROUTER_HIDDEN:-256}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SCALE_LOSS_WEIGHT="${SCALE_LOSS_WEIGHT:-0.03}"
COSINE_LOSS_WEIGHT="${COSINE_LOSS_WEIGHT:-0.003}"
DEVICE="${DEVICE:-cuda}"
MARGIN=${MARGIN:--0}

WEIGHT_KAPPA="${WEIGHT_KAPPA:-10}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DTYPE="${DTYPE:-float16}"

REUSE_FIG4_SEED42="${REUSE_FIG4_SEED42:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
SUMMARY_DIR="$FIG4A_DIR/summaries"
mkdir -p "$SUMMARY_DIR"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  configure_model "$MODEL_KEY"

  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}.npz"
  if [[ ! -s "$FEATURE_PATH" ]]; then
    echo "Missing feature file for $MODEL: $FEATURE_PATH" >&2
    echo "Extract features for this model before running Fig. 4a." >&2
    exit 1
  fi

  SUMMARY_JSON="$SUMMARY_DIR/${MODEL}_fold0.json"
  rm -f "$SUMMARY_JSON"

  echo
  echo "################################################################"
  echo "Fig. 4a | model=$MODEL | layer=$LAYER | fold=$FOLD_IDX | seeds=${SEED_VALUES[*]}"
  echo "################################################################"

  for SEED in "${SEED_VALUES[@]}"; do
    for VAL_FRACTION in "${VAL_FRACTIONS[@]}"; do
      FRACTION_TAG="${VAL_FRACTION/./p}"
      CHECKPOINT_PATH="$FIG4A_DIR/checkpoints/$MODEL/fold0/${MODEL}_layer${LAYER}_fold0_seed${SEED}_val${FRACTION_TAG}.npz"
      EVAL_JSON="$FIG4A_DIR/results/$MODEL/fold0/${MODEL}_layer${LAYER}_fold0_seed${SEED}_val${FRACTION_TAG}_eval.json"
      mkdir -p "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$EVAL_JSON")"

      # Fig. 4 already trained/evaluated fold0 with seed 42. Reuse those
      # immutable artifacts when requested, while keeping the Fig. 4a summary
      # independent and explicitly recording seed=42.
      SOURCE_CHECKPOINT="$FIG4A_DIR/../fig4/checkpoints/$MODEL/fold0/${MODEL}_layer${LAYER}_fold0_val${FRACTION_TAG}.npz"
      SOURCE_EVAL="$FIG4A_DIR/../fig4/results/$MODEL/fold0/${MODEL}_layer${LAYER}_fold0_val${FRACTION_TAG}_eval.json"
      if [[ "$SEED" == "42" && "$REUSE_FIG4_SEED42" == "1" && -s "$SOURCE_CHECKPOINT" && -s "$SOURCE_EVAL" ]]; then
        CHECKPOINT_PATH="$SOURCE_CHECKPOINT"
        EVAL_JSON="$SOURCE_EVAL"
        echo "========== val_fraction=$VAL_FRACTION | seed=$SEED | reuse Fig. 4 =========="
      elif [[ "$REUSE_EXISTING" == "1" && -s "$CHECKPOINT_PATH" && -s "$EVAL_JSON" ]]; then
        echo "========== val_fraction=$VAL_FRACTION | seed=$SEED | reuse Fig. 4a artifacts =========="
      else
        echo
        echo "========== val_fraction=$VAL_FRACTION | seed=$SEED | train =========="
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

        echo "========== val_fraction=$VAL_FRACTION | seed=$SEED | evaluate =========="
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
          --margin "$MARGIN" \
          --save_details
        test -s "$EVAL_JSON"
      fi

      "$PYTHON_BIN" "$FIG4A_DIR/collect_mc1.py" \
        --val-fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        --model "$MODEL" \
        --model-label "$MODEL_LABEL" \
        --layer "$LAYER" \
        --fold "$FOLD_IDX" \
        --num-folds "$NUM_FOLDS" \
        --eval-json "$EVAL_JSON" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$SUMMARY_JSON"
    done
  done
  echo "Saved summary: $SUMMARY_JSON"
done

echo
echo "Fig. 4a complete. Fold0-only summaries: $SUMMARY_DIR"

if [[ "${AUTO_PLOT:-1}" == "1" ]]; then
  SEEDS="${SEED_VALUES[*]}" bash "$FIG4A_DIR/plot.sh"
fi
