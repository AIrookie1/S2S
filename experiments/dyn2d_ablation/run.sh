#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn2d/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/hyperparams/dyn2d/experiments/dyn2d_ablation.sh"
source "$REPO_ROOT/script/common.sh"

# This experiment is intentionally fixed to the requested setting: Llama/fold0.
FOLD_IDX="$DYN2D_ABLATION_FOLD"
GPU="${GPU:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"
RUN_TRAIN="${RUN_TRAIN:-auto}"
AUTO_PLOT="${AUTO_PLOT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/dyn2d_ablation}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoints/${MODEL}_l${LAYER}_fold0_dynMLP_${POSITIVE_MODE}.npz}"
mkdir -p "$OUTPUT_DIR" "$(dirname "$CHECKPOINT_PATH")"

case "$RUN_EXTRACT" in auto|0|1) ;; *) echo "RUN_EXTRACT must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$RUN_TRAIN" in auto|0|1) ;; *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;; esac

if [[ "$RUN_EXTRACT" == 1 || ( "$RUN_EXTRACT" == auto && ! -s "$FEATURE_PATH" ) ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/extract_features.py "$MODEL" \
    --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$(dirname "$FEATURE_PATH")" \
    --output_name "$(basename "$FEATURE_PATH")" --batch_size "$EXTRACT_BATCH_SIZE" \
    --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
fi
[[ -s "$FEATURE_PATH" ]] || { echo "Missing feature file: $FEATURE_PATH" >&2; exit 1; }

if [[ "$RUN_TRAIN" == 1 || ( "$RUN_TRAIN" == auto && ! -s "$CHECKPOINT_PATH" ) ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/train_dyn_mlp.py \
    --feature_path "$FEATURE_PATH" --positive_mode "$POSITIVE_MODE" \
    --output_path "$CHECKPOINT_PATH" --num_folds "$NUM_FOLDS" --fold_idx 0 \
    --val_fraction "$VAL_FRACTION" --mlp_hidden "$MLP_HIDDEN" --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" --seed "$SEED"
fi
[[ -s "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }

STRENGTHS="${DYN2D_ABLATION_STRENGTHS[*]}"
SLERPS="${DYN2D_ABLATION_SLERPS[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$HERE/evaluate_grid.py" "$MODEL" \
  --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER" \
  --eval_split "$EVAL_SPLIT" --strengths "$STRENGTHS" --slerps "$SLERPS" \
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --preset "$EVAL_PRESET" \
  --output_dir "$OUTPUT_DIR" ${EXTRA_ARGS:-}

if [[ "$AUTO_PLOT" == 1 ]]; then
  OUTPUT_DIR="$OUTPUT_DIR" bash "$HERE/plot.sh"
fi
