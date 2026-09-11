#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODEL_KEY="${MODEL_KEY:-LLAMA}"; MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/hyperparams/dyn3d_sga/experiments/steering_vector_ablation.sh"
# Preserve Dyn3D model identity, then load the uploaded Spherical-Steering
# parameters for the same model (kappa / alpha / beta stay exactly as provided).
DYN_MODEL="$MODEL"; DYN_MODEL_DIR="$MODEL_DIR"; DYN_DTYPE="$DTYPE"; DYN_DEVICE="$DEVICE"; DYN_SEED="$SEED"
source "$REPO_ROOT/hyperparams/spherical_steering/$MODEL_KEY/truthfulqa.sh"
SPH_KAPPA="$KAPPA"; SPH_ALPHA="$ALPHA"; SPH_BETA="$BETA"
MODEL="$DYN_MODEL"; MODEL_DIR="$DYN_MODEL_DIR"; DTYPE="$DYN_DTYPE"; DEVICE="$DYN_DEVICE"; SEED="$DYN_SEED"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-1}"
NUM_FOLDS="${NUM_FOLDS:-2}"
FOLDS="${FOLDS:-0 1}"
LAYERS="${LAYERS:-$LAYERS_DEFAULT}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
PREPARE_FOLDS="${PREPARE_FOLDS:-1}"
BASE_FEATURE_TEMPLATE="${BASE_FEATURE_TEMPLATE:-$FEATURE_DIR/${MODEL}_layer{layer}_dynSVD_pairs.npz}"
FEATURE_TEMPLATE="${FEATURE_TEMPLATE:-$FEATURE_DIR/${MODEL}_layer{layer}_dynSVD_pairs_fold{fold}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/experiment/steering_vector_ablation}"
mkdir -p "$OUTPUT_DIR"
read -r -a layer_args <<< "$LAYERS"
read -r -a fold_args <<< "$FOLDS"

for layer in "${layer_args[@]}"; do
  base="${BASE_FEATURE_TEMPLATE//\{layer\}/$layer}"
  if [[ "$RUN_EXTRACT" == "1" ]]; then
    mkdir -p "$(dirname "$base")"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "$MODEL" \
      --model_dir "$MODEL_DIR" --layer "$layer" --save_dir "$(dirname "$base")" \
      --output_name "$(basename "$base")" --batch_size "$EXTRACT_BATCH_SIZE" \
      --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
  fi
  if [[ "$PREPARE_FOLDS" == "1" ]]; then
    [[ -f "$base" ]] || { echo "Missing base feature: $base (set RUN_EXTRACT=1 to create it)" >&2; exit 1; }
    prep=("$PYTHON_BIN" experiments/steering_vector_ablation/prepare_fold_features.py --input "$base" --num_folds "$NUM_FOLDS" --folds "${fold_args[@]}")
    [[ "${OVERWRITE_FOLDS:-0}" == "1" ]] && prep+=(--overwrite)
    "${prep[@]}"
  fi
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" experiments/steering_vector_ablation/evaluate.py "$MODEL" \
  --model_dir "$MODEL_DIR" --feature_template "$FEATURE_TEMPLATE" \
  --layers "${layer_args[@]}" --folds "${fold_args[@]}" \
  --linear_alpha "$LINEAR_ALPHA" --spherical_kappa "$SPH_KAPPA" \
  --spherical_alpha "$SPH_ALPHA" --spherical_beta "$SPH_BETA" \
  --device "$DEVICE" --dtype "$DTYPE" --seed "$SEED" --output_dir "$OUTPUT_DIR" ${EXTRA_ARGS:-}
