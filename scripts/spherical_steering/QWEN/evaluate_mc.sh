#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/configs/spherical_steering/QWEN/truthfulqa.sh"
source "$REPO_ROOT/scripts/common.sh"
FEATURE_PATH="${FEATURE_PATH:-$FEATURE_DIR/spherical_steering/truthfulqa/${MODEL}_layer${LAYER}.npz}"
PROTO_DIR="${PROTO_DIR:-$PROTOTYPE_DIR/spherical_steering/truthfulqa}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/truthfulqa/spherical_steering/${MODEL}}"
mkdir -p "$(dirname "$FEATURE_PATH")" "$PROTO_DIR" "$RESULT_DIR"

FOLD_IDX="${FOLD_IDX:-0}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$PROTO_DIR/${MODEL}_layer${LAYER}_fold${FOLD_IDX}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_DIR/mc}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_a${ALPHA}_b${BETA}.json}"
mkdir -p "$OUTPUT_DIR"
if [[ ! -s "$PROTOTYPE_PATH" ]]; then
  echo "[ERROR] Missing prototype: $PROTOTYPE_PATH" >&2
  echo "[HINT] Generate features first, then run compute_prototypes.sh." >&2
  echo "       bash $SCRIPT_DIR/extract_features.sh" >&2
  echo "       bash $SCRIPT_DIR/compute_prototypes.sh" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" methods/spherical_steering/evaluate_mc.py "$MODEL"   --prototype_path "$PROTOTYPE_PATH" --model_dir "$MODEL_DIR" --layer "$LAYER"   --kappa "$KAPPA" --alpha "$ALPHA" --beta "$BETA" --seed "$SEED"   --output_path "$OUTPUT_PATH" --save_details ${EXTRA_ARGS:-}
test -s "$OUTPUT_PATH"
