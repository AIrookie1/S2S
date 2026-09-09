#!/usr/bin/env bash
set -euo pipefail

# Fig. 7: token-level ternary density of adaptive (alpha, beta, gamma) weights.
# The model/fold/checkpoint are configurable; defaults reuse the existing
# Fig. 5 Llama fold0 rank-40 checkpoint.

GPU="${GPU:-3}"
FIG7_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FIG7_DIR/../../script/common.sh"
source "$FIG7_DIR/../model_config.sh"

MODEL_KEY="${MODEL_KEY:-llama}"
configure_model "$MODEL_KEY"
FOLD_IDX="${FOLD_IDX:-0}"
NUM_FOLDS="${NUM_FOLDS:-2}"
SEED="${SEED:-42}"
SVD_RANK="${SVD_RANK:-40}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DTYPE="${DTYPE:-float16}"
WEIGHTS_BINS="${WEIGHTS_BINS:-80}"
WEIGHT_KAPPAS="${WEIGHT_KAPPAS:-0.1 1 5 10 15 20 100}"

CHECKPOINT="${CHECKPOINT:-$FIG7_DIR/../fig5/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_layer${LAYER}_fold${FOLD_IDX}_rank${SVD_RANK}.npz}"
DATA_DIR="${DATA_DIR:-$FIG7_DIR/data}"
FIGURE_DIR="${FIGURE_DIR:-$FIG7_DIR/figures}"
mkdir -p "$DATA_DIR" "$FIGURE_DIR"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  echo "Set CHECKPOINT=/path/to/checkpoint.npz or run the Fig. 5 training first." >&2
  exit 1
fi

read -r -a KAPPAS <<< "$WEIGHT_KAPPAS"
for WEIGHT_KAPPA in "${KAPPAS[@]}"; do
  TAG="${WEIGHT_KAPPA/./p}"
  EVAL_JSON="$DATA_DIR/${MODEL}_fold${FOLD_IDX}_kappa${TAG}_eval.json"
  WEIGHTS_JSON="$DATA_DIR/${MODEL}_fold${FOLD_IDX}_kappa${TAG}_weights.json"
  if [[ "${SKIP_EVAL:-0}" == "1" && -s "$WEIGHTS_JSON" ]]; then
    echo "========== kappa=$WEIGHT_KAPPA | reuse density data =========="
    continue
  fi

  echo "========== kappa=$WEIGHT_KAPPA | TruthfulQA evaluation + weight histogram =========="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py \
    --method-dir dyn3d_sga \
    --entry-file evaluate_3d_dyn.py \
    -- \
    "$MODEL" \
    --checkpoint_path "$CHECKPOINT" \
    --model_dir "$MODEL_DIR" \
    --layer "$LAYER" \
    --eval_split "$EVAL_SPLIT" \
    --weight_kappa "$WEIGHT_KAPPA" \
    --device cuda \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --output_path "$EVAL_JSON" \
    --weights_output_path "$WEIGHTS_JSON" \
    --weights_bins "$WEIGHTS_BINS" \
    --save_details
  test -s "$EVAL_JSON"
  test -s "$WEIGHTS_JSON"
done

if [[ "${AUTO_PLOT:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$FIG7_DIR/plot_fig7.py" \
    --input-dir "$DATA_DIR" \
    --output-dir "$FIGURE_DIR" \
    --model-label "$MODEL_LABEL"
fi

echo "Fig. 7 complete: $FIGURE_DIR"
