#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

FAMILY="${FAMILY:-LLAMA}"; FAMILY="${FAMILY^^}"
case "$FAMILY" in LLAMA|QWEN) ;; *) echo "FAMILY must be LLAMA or QWEN" >&2; exit 2;; esac
source "$ROOT/scripts/common.sh"

# L2S defaults.
source "$ROOT/configs/l2s_linear/$FAMILY/truthfulqa.sh"
L2S_MODEL="$MODEL"; L2S_MODEL_DIR="$MODEL_DIR"; L2S_LAYER="$LAYER"; L2S_FOLD="$FOLD_IDX"
L2S_NUM_FOLDS="$NUM_FOLDS"; L2S_VAL="$VAL_FRACTION"; L2S_HIDDEN="$MLP_HIDDEN"
L2S_EPOCHS="$EPOCHS"; L2S_BATCH="$BATCH_SIZE"; L2S_LR="$LEARNING_RATE"
L2S_SEED="$SEED"; L2S_DEVICE="$DEVICE"; L2S_DTYPE="$DTYPE"; L2S_PRESET="$EVAL_PRESET"
LINEAR_ALPHA="$ALPHA"
unset ALPHA KAPPA BETA
source "$ROOT/configs/l2s_slerp/$FAMILY/truthfulqa.sh"
SLERP_KAPPA="$KAPPA"; SLERP_ALPHA="$ALPHA"; SLERP_BETA="$BETA"

# S2S-L-Static defaults.
source "$ROOT/configs/S2S_L/$FAMILY/truthfulqa.sh"
S2SL_STRENGTH="$STRENGTH"; S2SL_T="$SLERP_T"
S2S_POSITIVE_MODE="$POSITIVE_MODE"

# S2S-R defaults.
source "$ROOT/configs/S2S_R/$FAMILY/truthfulqa.sh"
S2SR_KAPPA="$WEIGHT_KAPPA"; S2SR_MARGIN="$MARGIN"

MODEL="${MODEL_OVERRIDE:-$L2S_MODEL}"
MODEL_DIR="${MODEL_DIR_OVERRIDE:-$L2S_MODEL_DIR}"
LAYER="${LAYER_OVERRIDE:-$L2S_LAYER}"
FOLD_IDX="${FOLD_IDX_OVERRIDE:-$L2S_FOLD}"
GPU="${GPU:-0}"
NOISE_LEVELS="${NOISE_LEVELS:-1.0 2.0 3.0 4.0 5.0}"
NOISE_SEED="${NOISE_SEED:-$L2S_SEED}"
RUN_TRAIN="${RUN_TRAIN:-auto}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/experiment/fig5_target_noise/$MODEL/fold${FOLD_IDX}}"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
S2S_FEATURE_PATH="${S2S_FEATURE_PATH:-$FEATURE_PATH}"
[[ -s "$FEATURE_PATH" ]] || FEATURE_PATH="$S2S_FEATURE_PATH"
mkdir -p "$RESULT_DIR"

case "$RUN_TRAIN" in
  auto|0|1) ;;
  *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;;
esac

# Use one canonical numeric representation for both the writer and collector:
# 1.0 -> 1, 1.5 -> 1p5, etc.
noise_tag() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}".replace(".", "p"), end="")
PY
}

# In auto mode, only reuse checkpoints that are explicitly marked as fixed
# target-noise training checkpoints with the requested sigma and noise seed.
checkpoint_matches_target_noise() {
  local path="$1"
  local sigma="$2"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$sigma" "$NOISE_SEED" <<'PY'
import sys
import numpy as np

path, sigma_text, seed_text = sys.argv[1:]
expected_sigma = float(sigma_text)
expected_seed = int(seed_text)
try:
    with np.load(path, allow_pickle=False) as checkpoint:
        required = {"target_noise_sigma", "noise_seed", "target_noise_sampling"}
        if not required.issubset(checkpoint.files):
            raise ValueError("missing target-noise metadata")
        actual_sigma = float(np.asarray(checkpoint["target_noise_sigma"]).item())
        actual_seed = int(np.asarray(checkpoint["noise_seed"]).item())
        sampling = str(np.asarray(checkpoint["target_noise_sampling"]).item())
        if not np.isclose(actual_sigma, expected_sigma, rtol=0.0, atol=1e-12):
            raise ValueError("target_noise_sigma mismatch")
        if actual_seed != expected_seed:
            raise ValueError("noise_seed mismatch")
        if "fixed across all epochs and batches" not in sampling:
            raise ValueError("target noise is not marked fixed")
except Exception:
    sys.exit(1)
PY
}

need_train() {
  local path="$1"
  local sigma="$2"
  case "$RUN_TRAIN" in
    1) return 0 ;;
    0) return 1 ;;
    auto)
      if checkpoint_matches_target_noise "$path" "$sigma"; then
        return 1
      fi
      return 0
      ;;
  esac
}

for SIGMA in $NOISE_LEVELS; do
  TAG="$(noise_tag "$SIGMA")"
  OUT="$RESULT_DIR/noise_$TAG"; mkdir -p "$OUT"
  LCP="$OUT/l2s_linear.npz"; SCP="$OUT/l2s_slerp.npz"
  D2CP="$OUT/s2s_l_static.npz"; D3CP="$OUT/s2s_r.npz"

  if need_train "$LCP" "$SIGMA"; then
    "$PYTHON_BIN" methods/l2s_linear/train_l2s.py --feature_path "$FEATURE_PATH" --output_path "$LCP" \
      --num_folds "$L2S_NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$L2S_VAL" \
      --mlp_hidden "$L2S_HIDDEN" --epochs "$L2S_EPOCHS" --batch_size "$L2S_BATCH" \
      --learning_rate "$L2S_LR" --device "$L2S_DEVICE" --seed "$L2S_SEED" \
      --target_noise_sigma "$SIGMA" --noise_seed "$NOISE_SEED"
  fi
  if need_train "$SCP" "$SIGMA"; then
    "$PYTHON_BIN" methods/l2s_slerp/train_l2s.py --feature_path "$FEATURE_PATH" --output_path "$SCP" \
      --num_folds "$L2S_NUM_FOLDS" --fold_idx "$FOLD_IDX" --val_fraction "$L2S_VAL" \
      --mlp_hidden "$L2S_HIDDEN" --epochs "$L2S_EPOCHS" --batch_size "$L2S_BATCH" \
      --learning_rate "$L2S_LR" --device "$L2S_DEVICE" --seed "$L2S_SEED" \
      --target_noise_sigma "$SIGMA" --noise_seed "$NOISE_SEED"
  fi

  # S2S_L_Static and S2S_R use the same direct-vector MLP objective, but are
  # trained independently here so every method has an auditable checkpoint.
  for SPEC in "S2S_L_Static:$D2CP" "S2S_R:$D3CP"; do
    CP="${SPEC#*:}"
    if need_train "$CP" "$SIGMA"; then
      "$PYTHON_BIN" methods/S2S_R/train_predictor.py --feature_path "$S2S_FEATURE_PATH" --output_path "$CP" \
        --positive_mode "$S2S_POSITIVE_MODE" --num_folds "$L2S_NUM_FOLDS" --fold_idx "$FOLD_IDX" \
        --val_fraction "$L2S_VAL" --mlp_hidden "$L2S_HIDDEN" --epochs "$L2S_EPOCHS" \
        --batch_size "$L2S_BATCH" --learning_rate "$L2S_LR" --device "$L2S_DEVICE" \
        --seed "$L2S_SEED" --target_noise_sigma "$SIGMA" --noise_seed "$NOISE_SEED"
    fi
  done

  # This experiment studies training-target corruption only. Pin all generic
  # inference residual-noise controls to zero so evaluator defaults cannot
  # silently change the experiment definition in the future.
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" experiments/fig5_target_noise/evaluate_l2s_baselines.py "$MODEL" \
    --linear_checkpoint_path "$LCP" --slerp_checkpoint_path "$SCP" --model_dir "$MODEL_DIR" --layer "$LAYER" \
    --eval_split test --linear_alpha "$LINEAR_ALPHA" --slerp_kappa "$SLERP_KAPPA" \
    --slerp_alpha "$SLERP_ALPHA" --slerp_beta "$SLERP_BETA" --device "$L2S_DEVICE" --dtype "$L2S_DTYPE" \
    --seed "$L2S_SEED" --preset "$L2S_PRESET" --residual_noise_sigma 0.0 \
    --output_dir "$OUT" --output_stem l2s

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/S2S_L/evaluate_S2S_L.py "$MODEL" \
    --checkpoint_path "$D2CP" --model_dir "$MODEL_DIR" --layer "$LAYER" --eval_split test \
    --strength "$S2SL_STRENGTH" --slerp_t "$S2SL_T" --method S2S_L_Static \
    --device "$L2S_DEVICE" --dtype "$L2S_DTYPE" --seed "$L2S_SEED" --preset "$L2S_PRESET" \
    --residual_noise_sigma 0.0 --output_path "$OUT/s2s_l_static.json"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/S2S_R/evaluate_S2S_R.py "$MODEL" \
    --checkpoint_path "$D3CP" --model_dir "$MODEL_DIR" --layer "$LAYER" --eval_split test \
    --weight_kappa "$S2SR_KAPPA" --margin "$S2SR_MARGIN" --method S2S_R \
    --device "$L2S_DEVICE" --dtype "$L2S_DTYPE" --seed "$L2S_SEED" --preset "$L2S_PRESET" \
    --residual_noise_sigma 0.0 --output_path "$OUT/s2s_r.json"
done

"$PYTHON_BIN" "$HERE/collect.py" --input_dir "$RESULT_DIR" --noise_levels $NOISE_LEVELS
"$PYTHON_BIN" "$HERE/fig5_plot.py" --input "$RESULT_DIR/mc1_by_noise.csv" \
  --output_prefix "$RESULT_DIR/fig5_target_noise"
