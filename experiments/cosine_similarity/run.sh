#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_KEY="${MODEL_KEY:-LLAMA}"
MODEL_KEY="${MODEL_KEY^^}"
case "$MODEL_KEY" in
  LLAMA) ;;
  *)
    echo "The integrated SADI-hidden cosine pipeline currently supports MODEL_KEY=LLAMA only." >&2
    exit 2
    ;;
esac

# Dyn3D config supplies the common feature/model identity and inference kappa.
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
NUM_FOLDS="${NUM_FOLDS:-2}"
FOLDS="${FOLDS:-0 1}"
LAYER="${LAYER:-$DEFAULT_LAYER}"
RUN_EXTRACT="${RUN_EXTRACT:-auto}"
RUN_TRAIN="${RUN_TRAIN:-auto}"
DRY_RUN="${DRY_RUN:-0}"
WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"

# Uploaded SADI hidden-branch defaults.
SADI_COUNTS="${SADI_COUNTS:-1 2 4 6 8 10 12 14 16 18 20}"
SADI_STRENGTHS="${SADI_STRENGTHS:-5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80}"

case "$RUN_EXTRACT" in auto|0|1) ;; *) echo "RUN_EXTRACT must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$RUN_TRAIN" in auto|0|1) ;; *) echo "RUN_TRAIN must be auto, 0, or 1" >&2; exit 2 ;; esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac

FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/cosine_similarity}"
BATCH_SIZE_COS="${BATCH_SIZE_COS:-4096}"
COS_DEVICE="${COS_DEVICE:-$DEVICE}"

# Bash variables cannot safely contain a literal {fold} inside a ${VAR:-...}
# default expansion, so build checkpoint templates in two steps.
L2S_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/l2s/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_l2s_fold"
L2S_TEMPLATE_DEFAULT+='{fold}.npz'
L2S_CHECKPOINT_TEMPLATE="${L2S_CHECKPOINT_TEMPLATE:-$L2S_TEMPLATE_DEFAULT}"

DYN_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynSVD_fold"
DYN_TEMPLATE_DEFAULT+="{fold}_${POSITIVE_MODE}.npz"
DYN3D_CHECKPOINT_TEMPLATE="${DYN3D_CHECKPOINT_TEMPLATE:-$DYN_TEMPLATE_DEFAULT}"

DYNLEARNED_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/dynlearnedslerp/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynLearnedSLERP_fold"
DYNLEARNED_TEMPLATE_DEFAULT+="{fold}_cartesian.npz"
DYNLEARNEDSLERP_CHECKPOINT_TEMPLATE="${DYNLEARNEDSLERP_CHECKPOINT_TEMPLATE:-$DYNLEARNED_TEMPLATE_DEFAULT}"

DYNMLP_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/dynmlp/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynMLP_fold"
DYNMLP_TEMPLATE_DEFAULT+="{fold}_cartesian.npz"
DYNMLP_CHECKPOINT_TEMPLATE="${DYNMLP_CHECKPOINT_TEMPLATE:-$DYNMLP_TEMPLATE_DEFAULT}"

SADI_TEMPLATE_DEFAULT="$CHECKPOINT_ROOT/sadi/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_sadi_hidden_fold"
SADI_TEMPLATE_DEFAULT+='{fold}.npz'
SADI_CHECKPOINT_TEMPLATE="${SADI_CHECKPOINT_TEMPLATE:-$SADI_TEMPLATE_DEFAULT}"

read -r -a fold_args <<< "$FOLDS"
read -r -a sadi_count_args <<< "$SADI_COUNTS"
read -r -a sadi_strength_args <<< "$SADI_STRENGTHS"
if [[ "${#fold_args[@]}" -eq 0 ]]; then
  echo "No folds selected. Set FOLDS, e.g. FOLDS='0 1'." >&2
  exit 2
fi
if [[ "${#sadi_count_args[@]}" -eq 0 || "${#sadi_strength_args[@]}" -eq 0 ]]; then
  echo "SADI_COUNTS and SADI_STRENGTHS must be non-empty." >&2
  exit 2
fi

render_fold() {
  local template="$1"
  local fold="$2"
  printf '%s\n' "${template//\{fold\}/$fold}"
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

feature_has_sadi_fields() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
required = {
    "sadi_mlp_activations",
    "sadi_mlp_z_pos_raw",
    "sadi_mlp_z_neg_raw",
    "sadi_mlp_v_T",
}
try:
    with np.load(path, allow_pickle=False) as data:
        ok = required.issubset(data.files)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

extract_features() {
  echo "[cosine/extract] model=$MODEL layer=$LAYER -> $FEATURE_PATH"
  mkdir -p "$(dirname "$FEATURE_PATH")"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    method/dyn3d_sga/extract_features.py "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --layer "$LAYER" \
    --save_dir "$(dirname "$FEATURE_PATH")" \
    --output_name "$(basename "$FEATURE_PATH")" \
    --batch_size "$EXTRACT_BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --split_seed "$SEED"
}

need_extract=0
if [[ ! -s "$FEATURE_PATH" ]]; then
  need_extract=1
elif ! feature_has_sadi_fields "$FEATURE_PATH"; then
  echo "[cosine] Existing feature file lacks integrated SADI MLP fields: $FEATURE_PATH"
  need_extract=1
fi

if [[ "$RUN_EXTRACT" == "1" || ( "$RUN_EXTRACT" == "auto" && "$need_extract" == "1" ) ]]; then
  extract_features
elif [[ "$need_extract" == "1" && "$DRY_RUN" != "1" ]]; then
  echo "Feature file is missing or lacks SADI fields: $FEATURE_PATH" >&2
  echo "Set RUN_EXTRACT=1/auto to regenerate it with the integrated extractor." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]] && ! feature_has_sadi_fields "$FEATURE_PATH"; then
  echo "Feature extraction completed but required SADI fields are still missing: $FEATURE_PATH" >&2
  exit 1
fi

train_dyn3d_fold() {
  local fold="$1"
  local output="$2"
  echo "[cosine/train Dyn3D] fold=$fold -> $output"
  mkdir -p "$(dirname "$output")"
  run_cmd env \
    GPU="$GPU" \
    FOLD_IDX="$fold" \
    LAYER="$LAYER" \
    NUM_FOLDS="$NUM_FOLDS" \
    VAL_FRACTION="$VAL_FRACTION" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    POSITIVE_MODE="cartesian" \
    FEATURE_PATH="$FEATURE_PATH" \
    OUTPUT_PATH="$output" \
    "$REPO_ROOT/script/dyn3d_sga/$MODEL_KEY/run_3d_train.sh"
}

train_dynlearnedslerp_fold() {
  local fold="$1"
  local output="$2"
  echo "[cosine/train DynLearnedSLERP] fold=$fold -> $output"
  mkdir -p "$(dirname "$output")"
  run_cmd env \
    GPU="$GPU" \
    FOLD_IDX="$fold" \
    LAYER="$LAYER" \
    NUM_FOLDS="$NUM_FOLDS" \
    VAL_FRACTION="$VAL_FRACTION" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    POSITIVE_MODE="cartesian" \
    FEATURE_PATH="$FEATURE_PATH" \
    OUTPUT_PATH="$output" \
    "$REPO_ROOT/script/dynlearnedslerp/$MODEL_KEY/run_train.sh"
}

train_dynmlp_fold() {
  local fold="$1"
  local output="$2"
  echo "[cosine/train dynMLP] fold=$fold -> $output"
  mkdir -p "$(dirname "$output")"
  run_cmd env \
    GPU="$GPU" \
    FOLD_IDX="$fold" \
    LAYER="$LAYER" \
    NUM_FOLDS="$NUM_FOLDS" \
    VAL_FRACTION="$VAL_FRACTION" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    POSITIVE_MODE="cartesian" \
    FEATURE_PATH="$FEATURE_PATH" \
    OUTPUT_PATH="$output" \
    "$REPO_ROOT/script/dynmlp/$MODEL_KEY/run_train.sh"
}

train_l2s_fold() {
  local fold="$1"
  local output="$2"
  echo "[cosine/train L2S] fold=$fold -> $output"
  mkdir -p "$(dirname "$output")"
  run_cmd env \
    GPU="$GPU" \
    FOLD_IDX="$fold" \
    LAYER="$LAYER" \
    NUM_FOLDS="$NUM_FOLDS" \
    VAL_FRACTION="$VAL_FRACTION" \
    SEED="$SEED" \
    DEVICE="$DEVICE" \
    FEATURE_PATH="$FEATURE_PATH" \
    OUTPUT_PATH="$output" \
    "$REPO_ROOT/script/l2s/$MODEL_KEY/run_train.sh"
}

sadi_checkpoint_matches_metric() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
try:
    with np.load(path, allow_pickle=False) as data:
        if "cosine_gt_mode" not in data.files:
            ok = False
        else:
            mode = str(np.asarray(data["cosine_gt_mode"]).item())
            ok = mode == "mean_all_cartesian_pairs"
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

train_sadi_fold() {
  local fold="$1"
  local output="$2"
  echo "[cosine/train SADI-hidden] fold=$fold -> $output"
  mkdir -p "$(dirname "$output")"
  run_cmd "$PYTHON_BIN" method/sadi/train_sadi.py \
    --feature_path "$FEATURE_PATH" \
    --output_path "$output" \
    --num_folds "$NUM_FOLDS" \
    --fold_idx "$fold" \
    --val_fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    --counts "${sadi_count_args[@]}" \
    --strengths "${sadi_strength_args[@]}"
}

for fold in "${fold_args[@]}"; do
  if (( fold < 0 || fold >= NUM_FOLDS )); then
    echo "Fold $fold is outside [0, $((NUM_FOLDS - 1))]" >&2
    exit 2
  fi

  l2s_ckpt="$(render_fold "$L2S_CHECKPOINT_TEMPLATE" "$fold")"
  dyn_ckpt="$(render_fold "$DYN3D_CHECKPOINT_TEMPLATE" "$fold")"
  dynlearned_ckpt="$(render_fold "$DYNLEARNEDSLERP_CHECKPOINT_TEMPLATE" "$fold")"
  dynmlp_ckpt="$(render_fold "$DYNMLP_CHECKPOINT_TEMPLATE" "$fold")"
  sadi_ckpt="$(render_fold "$SADI_CHECKPOINT_TEMPLATE" "$fold")"

  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$l2s_ckpt" ) ]]; then
    train_l2s_fold "$fold" "$l2s_ckpt"
  elif [[ ! -s "$l2s_ckpt" && "$DRY_RUN" != "1" ]]; then
    echo "Missing L2S checkpoint: $l2s_ckpt" >&2
    exit 1
  fi

  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$dyn_ckpt" ) ]]; then
    train_dyn3d_fold "$fold" "$dyn_ckpt"
  elif [[ ! -s "$dyn_ckpt" && "$DRY_RUN" != "1" ]]; then
    echo "Missing Dyn3D checkpoint: $dyn_ckpt" >&2
    exit 1
  fi

  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$dynlearned_ckpt" ) ]]; then
    train_dynlearnedslerp_fold "$fold" "$dynlearned_ckpt"
  elif [[ ! -s "$dynlearned_ckpt" && "$DRY_RUN" != "1" ]]; then
    echo "Missing DynLearnedSLERP checkpoint: $dynlearned_ckpt" >&2
    exit 1
  fi

  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && ! -s "$dynmlp_ckpt" ) ]]; then
    train_dynmlp_fold "$fold" "$dynmlp_ckpt"
  elif [[ ! -s "$dynmlp_ckpt" && "$DRY_RUN" != "1" ]]; then
    echo "Missing dynMLP checkpoint: $dynmlp_ckpt" >&2
    exit 1
  fi

  sadi_needs_train=0
  if [[ ! -s "$sadi_ckpt" ]]; then
    sadi_needs_train=1
  elif ! sadi_checkpoint_matches_metric "$sadi_ckpt"; then
    echo "[cosine] Existing SADI checkpoint uses a different cosine GT metric: $sadi_ckpt"
    sadi_needs_train=1
  fi

  if [[ "$RUN_TRAIN" == "1" || ( "$RUN_TRAIN" == "auto" && "$sadi_needs_train" == "1" ) ]]; then
    train_sadi_fold "$fold" "$sadi_ckpt"
  elif [[ "$sadi_needs_train" == "1" && "$DRY_RUN" != "1" ]]; then
    echo "Missing/incompatible SADI checkpoint: $sadi_ckpt" >&2
    echo "Set RUN_TRAIN=1/auto to retrain it with all-pairs mean cosine validation." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
cmd=(
  "$PYTHON_BIN" experiments/cosine_similarity/evaluate.py
  --feature_path "$FEATURE_PATH"
  --l2s_checkpoint_template "$L2S_CHECKPOINT_TEMPLATE"
  --dyn3d_checkpoint_template "$DYN3D_CHECKPOINT_TEMPLATE"
  --dynlearnedslerp_checkpoint_template "$DYNLEARNEDSLERP_CHECKPOINT_TEMPLATE"
  --dynmlp_checkpoint_template "$DYNMLP_CHECKPOINT_TEMPLATE"
  --sadi_checkpoint_template "$SADI_CHECKPOINT_TEMPLATE"
  --num_folds "$NUM_FOLDS"
  --folds "${fold_args[@]}"
  --weight_kappa "$WEIGHT_KAPPA"
   --batch_size "$BATCH_SIZE_COS"
  --device "$COS_DEVICE"
  --output_dir "$OUTPUT_DIR"
)

echo "[cosine/evaluate] output=$OUTPUT_DIR"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
