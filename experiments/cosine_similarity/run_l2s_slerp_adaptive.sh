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
FOLDS="${FOLDS:-0 1}"
# Match the current three-sphere TruthfulQA adaptive-weight default.
WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"   # auto | 0 | 1
RUN_TRAIN="${RUN_TRAIN:-auto}"       # auto | 0 | 1
COS_DEVICE="${COS_DEVICE:-cpu}"
COS_BATCH_SIZE="${COS_BATCH_SIZE:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/cosine_similarity/l2s_slerp}"
OUTPUT_STEM="${OUTPUT_STEM:-${MODEL}_layer${LAYER}_l2s_slerp_adaptive}"

case "$RUN_EXTRACT" in auto|0|1) ;; *) echo "RUN_EXTRACT must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$RUN_TRAIN" in auto|0|1) ;; *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;; esac

FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
if [[ ! -s "$FEATURE_PATH" && -s "$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_pairs.npz" ]]; then
  FEATURE_PATH="$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_pairs.npz"
fi

L2S_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/l2s/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_l2s_fold"
L2S_TEMPLATE_DEFAULT+='{fold}.npz'
L2S_CHECKPOINT_TEMPLATE="${L2S_CHECKPOINT_TEMPLATE:-$L2S_TEMPLATE_DEFAULT}"

read -r -a fold_args <<< "$FOLDS"
[[ ${#fold_args[@]} -gt 0 ]] || { echo "FOLDS is empty" >&2; exit 2; }

render_fold() {
  local template="$1" fold="$2"
  printf '%s\n' "${template//\{fold\}/$fold}"
}

if [[ ! -s "$FEATURE_PATH" ]]; then
  if [[ "$RUN_EXTRACT" == "1" || "$RUN_EXTRACT" == "auto" ]]; then
    echo "[cos] extracting raw TruthfulQA features -> $FEATURE_PATH"
    mkdir -p "$(dirname "$FEATURE_PATH")"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "$MODEL" \
      --model_dir "$MODEL_DIR" \
      --layer "$LAYER" \
      --save_dir "$(dirname "$FEATURE_PATH")" \
      --output_name "$(basename "$FEATURE_PATH")" \
      --batch_size "$EXTRACT_BATCH_SIZE" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --split_seed "$SEED"
  else
    echo "Missing feature file: $FEATURE_PATH" >&2
    exit 1
  fi
fi
test -s "$FEATURE_PATH"

for fold in "${fold_args[@]}"; do
  ckpt="$(render_fold "$L2S_CHECKPOINT_TEMPLATE" "$fold")"
  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$ckpt" ) ]]; then
    echo "[cos] training L2S fold=$fold -> $ckpt"
    GPU="$GPU" \
    FOLD_IDX="$fold" \
    NUM_FOLDS="$NUM_FOLDS" \
    LAYER="$LAYER" \
    FEATURE_PATH="$FEATURE_PATH" \
    OUTPUT_PATH="$ckpt" \
    "$REPO_ROOT/script/l2s/$MODEL_KEY/run_train.sh"
  elif [[ ! -s "$ckpt" ]]; then
    echo "Missing L2S checkpoint: $ckpt" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
echo "[cos] L2S baseline + adaptive L2S-SLERP"
echo "      adaptive t = w_residual / (w_mean + w_residual)"
echo "      weight_kappa = $WEIGHT_KAPPA"
echo "      folds = $FOLDS"
"$PYTHON_BIN" experiments/cosine_similarity/evaluate_l2s_slerp_adaptive.py \
  --feature_path "$FEATURE_PATH" \
  --l2s_checkpoint_template "$L2S_CHECKPOINT_TEMPLATE" \
  --folds "${fold_args[@]}" \
  --num_folds "$NUM_FOLDS" \
  --weight_kappa "$WEIGHT_KAPPA" \
  --batch_size "$COS_BATCH_SIZE" \
  --device "$COS_DEVICE" \
  --output_dir "$OUTPUT_DIR" \
  --output_stem "$OUTPUT_STEM"

echo "[cos] done"
echo "JSON: $OUTPUT_DIR/$OUTPUT_STEM.json"
echo "CSV : $OUTPUT_DIR/$OUTPUT_STEM.csv"
