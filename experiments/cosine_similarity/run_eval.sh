#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_KEY="${MODEL_KEY:-LLAMA}"
MODEL_KEY="${MODEL_KEY^^}"
case "$MODEL_KEY" in
  LLAMA|QWEN) ;;
  *) echo "MODEL_KEY must be LLAMA or QWEN" >&2; exit 2 ;;
esac

source "$REPO_ROOT/hyperparams/l2s/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
ALPHA="${ALPHA:-1.0}"
# Same adaptive-weight sharpness as the current three-sphere TruthfulQA setup.
WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"
RUN_TRAIN="${RUN_TRAIN:-auto}"       # auto | 0 | 1
SAVE_DETAILS="${SAVE_DETAILS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/truthfulqa_l2s_slerp/$MODEL/fold${FOLD_IDX}}"
OUTPUT_STEM="${OUTPUT_STEM:-${MODEL}_layer${LAYER}_fold${FOLD_IDX}_alpha${ALPHA}_l2s_vs_l2s_slerp_adaptive}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/l2s/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_l2s_fold${FOLD_IDX}.npz}"

case "$RUN_TRAIN" in auto|0|1) ;; *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$SAVE_DETAILS" in 0|1) ;; *) echo "SAVE_DETAILS must be 0 or 1" >&2; exit 2 ;; esac

FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
if [[ ! -s "$FEATURE_PATH" && -s "$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_pairs.npz" ]]; then
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_pairs.npz"
fi

if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$CHECKPOINT_PATH" ) ]]; then
  if [[ ! -s "$FEATURE_PATH" ]]; then
    echo "Missing feature file for L2S training: $FEATURE_PATH" >&2
    echo "Run feature extraction first, or set FEATURE_PATH." >&2
    exit 1
  fi
  echo "[TruthfulQA] training L2S fold=$FOLD_IDX -> $CHECKPOINT_PATH"
  GPU="$GPU" \
  FOLD_IDX="$FOLD_IDX" \
  NUM_FOLDS="$NUM_FOLDS" \
  LAYER="$LAYER" \
  FEATURE_PATH="$FEATURE_PATH" \
  OUTPUT_PATH="$CHECKPOINT_PATH" \
  "$REPO_ROOT/script/l2s/$MODEL_KEY/run_train.sh"
fi

test -s "$CHECKPOINT_PATH"
if [[ ! -s "$FEATURE_PATH" ]]; then
  echo "Missing feature file needed to compute geometry-train v_avg: $FEATURE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

ARGS=(
  "$MODEL"
  --checkpoint_path "$CHECKPOINT_PATH"
  --feature_path "$FEATURE_PATH"
  --model_dir "$MODEL_DIR"
  --layer "$LAYER"
  --eval_split "$EVAL_SPLIT"
  --alpha "$ALPHA"
  --weight_kappa "$WEIGHT_KAPPA"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --preset "$EVAL_PRESET"
  --output_dir "$OUTPUT_DIR"
  --output_stem "$OUTPUT_STEM"
)
[[ "${FEW_SHOT:-0}" == 1 ]] && ARGS+=(--few_shot)
[[ "${NO_INSTRUCTION:-0}" == 1 ]] && ARGS+=(--no_instruction)
[[ "$SAVE_DETAILS" == 1 ]] && ARGS+=(--save_details)

echo "[TruthfulQA] additive L2S vs adaptive additive L2S-SLERP"
echo "             alpha=$ALPHA"
echo "             weight_kappa=$WEIGHT_KAPPA"
echo "             adaptive t = w_residual / (w_mean + w_residual)"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" experiments/truthfulqa/evaluate_l2s_vs_l2s_slerp.py "${ARGS[@]}"

echo "[TruthfulQA] done"
echo "JSON: $OUTPUT_DIR/$OUTPUT_STEM.json"
echo "CSV : $OUTPUT_DIR/$OUTPUT_STEM.csv"
