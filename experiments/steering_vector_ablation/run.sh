#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

MODEL_KEY="${MODEL_KEY:-LLAMA}"
MODEL_KEY="${MODEL_KEY^^}"
source "$REPO_ROOT/hyperparams/dyn3d_sga/$MODEL_KEY/truthfulqa.sh"
source "$REPO_ROOT/hyperparams/dyn3d_sga/experiments/steering_vector_ablation.sh"

source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-1}"
NUM_FOLDS="${NUM_FOLDS:-2}"
FOLDS="${FOLDS:-0 1}"
LAYERS="${LAYERS:-$LAYERS_DEFAULT}"

# This runner intentionally recomputes ONLY gt_neg_to_pos.
# Existing linear_shared_mean / spherical_shared_mean values in each layer JSON
# are preserved by evaluate.py when the new GT value is merged back in.

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
  printf '%s\n' "${template//\{layer\}/$layer}"
}

# GT-only rerun needs the already-extracted raw pair endpoints.  Do not extract
# or rewrite features here; fail early if any requested layer is missing.
for layer in "${layer_args[@]}"; do
  base="$(render_layer_template "$BASE_FEATURE_TEMPLATE" "$layer")"
  if [[ ! -f "$base" ]]; then
    echo "Missing base pair feature: $base" >&2
    echo "GT-only evaluation requires the existing *_dynSVD_pairs.npz file with z_pos_raw/z_neg_raw." >&2
    exit 1
  fi
  echo "[gt-only] reuse existing feature: $base"
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  experiments/steering_vector_ablation/evaluate.py "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --feature_template "$BASE_FEATURE_TEMPLATE" \
  --layers "${layer_args[@]}" \
  --folds "${fold_args[@]}" \
  --num_folds "$NUM_FOLDS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  ${EXTRA_ARGS:-}
