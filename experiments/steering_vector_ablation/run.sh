#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

MODEL_KEY="${MODEL_KEY:-LLAMA}"
MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/hyperparams/dyn3d_sga/experiments/steering_vector_ablation.sh"

# Preserve Dyn3D model identity, then load the uploaded Spherical-Steering
# parameters for the same model.
DYN_MODEL="$MODEL"
DYN_MODEL_DIR="$MODEL_DIR"
DYN_DTYPE="$DTYPE"
DYN_DEVICE="$DEVICE"
DYN_SEED="$SEED"

source "$REPO_ROOT/hyperparams/spherical_steering/$MODEL_KEY/truthfulqa.sh"
SPH_KAPPA="$KAPPA"
SPH_ALPHA="$ALPHA"
SPH_BETA="$BETA"

MODEL="$DYN_MODEL"
MODEL_DIR="$DYN_MODEL_DIR"
DTYPE="$DYN_DTYPE"
DEVICE="$DYN_DEVICE"
SEED="$DYN_SEED"

source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
NUM_FOLDS="${NUM_FOLDS:-2}"
FOLDS="${FOLDS:-0 1}"
LAYERS="${LAYERS:-$LAYERS_DEFAULT}"

# RUN_EXTRACT:
#   auto (default): reuse the single base pairs.npz; extract it only if missing.
#   0:              never extract; fail if the base pairs.npz is missing.
#   1:              force re-extraction of the base pairs.npz.
RUN_EXTRACT="${RUN_EXTRACT:-auto}"

# Remove the old duplicated *_fold0.npz / *_fold1.npz files produced by the
# previous version of this experiment.  The current version never writes them.
CLEAN_LEGACY_FOLD_FEATURES="${CLEAN_LEGACY_FOLD_FEATURES:-1}"

case "$RUN_EXTRACT" in
  auto|0|1) ;;
  *) echo "RUN_EXTRACT must be one of: auto, 0, 1" >&2; exit 2 ;;
esac

BASE_FEATURE_TEMPLATE="${BASE_FEATURE_TEMPLATE:-}"
if [[ -z "$BASE_FEATURE_TEMPLATE" ]]; then
  BASE_FEATURE_TEMPLATE="$FEATURE_DIR/${MODEL}_layer{layer}_dynSVD_pairs.npz"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/steering_vector_ablation}"
mkdir -p "$OUTPUT_DIR"

read -r -a layer_args <<< "$LAYERS"
read -r -a fold_args <<< "$FOLDS"

if [[ "${#layer_args[@]}" -eq 0 ]]; then
  echo "No layers selected. Set LAYERS, e.g. LAYERS='14'." >&2
  exit 2
fi

if [[ "${#fold_args[@]}" -eq 0 ]]; then
  echo "No folds selected. Set FOLDS, e.g. FOLDS='0 1'." >&2
  exit 2
fi

render_layer_template() {
  local template="$1"
  local layer="$2"
  local rendered="${template//\{layer\}/$layer}"
  printf '%s\n' "$rendered"
}

extract_base_feature() {
  local layer="$1"
  local base="$2"

  mkdir -p "$(dirname "$base")"
  echo "[extract] model=$MODEL layer=$layer -> $base"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    method/dyn3d_sga/extract_features.py "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --layer "$layer" \
    --save_dir "$(dirname "$base")" \
    --output_name "$(basename "$base")" \
    --batch_size "$EXTRACT_BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --split_seed "$SEED"
}

recover_broken_v2_feature_name() {
  local layer="$1"
  local base="$2"

  [[ -f "$base" ]] && return 0

  # Compatibility with the earlier bash-template bug.  The old script printed:
  #   <model>_layer{layer_dynSVD_pairs.npz}
  # Because that string does not actually end in ".npz", numpy may have written
  # an additional ".npz", producing:
  #   <model>_layer{layer_dynSVD_pairs.npz}.npz
  local broken_reported="$FEATURE_DIR/${MODEL}_layer{layer_dynSVD_pairs.npz}"
  local broken_appended="${broken_reported}.npz"
  local broken=""

  if [[ -f "$broken_reported" ]]; then
    broken="$broken_reported"
  elif [[ -f "$broken_appended" ]]; then
    broken="$broken_appended"
  else
    return 0
  fi

  # The feature extractor stores the true numeric layer inside the NPZ.  Only
  # migrate the malformed file when that metadata matches the requested layer.
  local stored_layer
  stored_layer="$(
    "$PYTHON_BIN" - "$broken" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as data:
    if "layer" not in data.files:
        print(-1)
    else:
        print(int(np.asarray(data["layer"]).item()))
PY
  )"

  if [[ "$stored_layer" == "$layer" ]]; then
    echo "[migrate] recovering previously extracted feature: $broken -> $base"
    mkdir -p "$(dirname "$base")"
    mv -- "$broken" "$base"
  else
    echo "[migrate] found malformed feature $broken but stored layer=$stored_layer; expected layer=$layer, leaving it untouched" >&2
  fi
}

cleanup_legacy_fold_features() {
  local base="$1"

  [[ "$CLEAN_LEGACY_FOLD_FEATURES" == "1" ]] || return 0

  local stem="$base"
  if [[ "$stem" == *.npz ]]; then
    stem="${stem%.npz}"
  fi

  local fold
  local legacy
  for fold in "${fold_args[@]}"; do
    legacy="${stem}_fold${fold}.npz"
    if [[ -f "$legacy" ]]; then
      echo "[cleanup] removing legacy duplicated fold feature: $legacy"
      rm -f -- "$legacy"
    fi
  done
}

for layer in "${layer_args[@]}"; do
  base="$(render_layer_template "$BASE_FEATURE_TEMPLATE" "$layer")"

  recover_broken_v2_feature_name "$layer" "$base"
  cleanup_legacy_fold_features "$base"

  if [[ "$RUN_EXTRACT" == "1" ]]; then
    extract_base_feature "$layer" "$base"
  elif [[ "$RUN_EXTRACT" == "auto" && ! -f "$base" ]]; then
    extract_base_feature "$layer" "$base"
  fi

  if [[ ! -f "$base" ]]; then
    echo "Missing base feature: $base" >&2
    echo "Use RUN_EXTRACT=auto/1 or provide the single *_dynSVD_pairs.npz file." >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  experiments/steering_vector_ablation/evaluate.py "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --feature_template "$BASE_FEATURE_TEMPLATE" \
  --layers "${layer_args[@]}" \
  --folds "${fold_args[@]}" \
  --num_folds "$NUM_FOLDS" \
  --linear_alpha "$LINEAR_ALPHA" \
  --spherical_kappa "$SPH_KAPPA" \
  --spherical_alpha "$SPH_ALPHA" \
  --spherical_beta "$SPH_BETA" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  ${EXTRA_ARGS:-}
