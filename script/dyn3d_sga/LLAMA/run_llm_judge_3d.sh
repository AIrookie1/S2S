#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=llama3.1-8B-Instruct
MODEL_DIR=/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct

LAYER=${LAYER:-14}
GPU=${GPU:-0}
FOLD_IDX=${FOLD_IDX:-0}

source "$SCRIPT_DIR/../../common.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

WEIGHT_KAPPA=${WEIGHT_KAPPA:-5}
MARGIN=${MARGIN:--0.3}
PRESET=${PRESET:-null}

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_mean.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$JUDGE_DIR/llama_truthfulqa}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_3d_fixed_router.json}"

# Cache contains:
#   1) full unsteered answer A0
#   2) Q+A0 target-layer last token id/text
#   3) Q+A0 target-layer last-token hidden state h_last
#
# It is intentionally checkpoint-independent, so a new router/checkpoint or
# WEIGHT_KAPPA can reuse the expensive Pass-1/2 model work.
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_base_probe_cache.pt}"

TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"

# Optional controls:
#   REBUILD_CACHE=1  -> discard/rebuild Pass-1/2 cache
#   CACHE_ONLY=1     -> only build/reuse Pass-1/2 cache; do not run Pass 3/judges
#   EXTRA_ARGS="..." -> any additional Python arguments, e.g. --max_samples 20
REBUILD_CACHE="${REBUILD_CACHE:-1}"
CACHE_ONLY="${CACHE_ONLY:-0}"

mkdir -p "$OUTPUT_DIR"
test -s "$CHECKPOINT_PATH"

PY_ARGS=(
  "$MODEL"
  --model_dir "$MODEL_DIR"
  --checkpoint_path "$CHECKPOINT_PATH"
  --layer "$LAYER"
  --output_dir "$OUTPUT_DIR"
  --output_path "$OUTPUT_PATH"
  --cache_path "$CACHE_PATH"
  --truth_judge_path "$TRUTH_JUDGE_PATH"
  --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$PRESET"
  --weight_kappa "$WEIGHT_KAPPA"
  --margin "$MARGIN"
)

if [[ "$REBUILD_CACHE" == "1" ]]; then
  PY_ARGS+=(--rebuild_cache)
fi

if [[ "$CACHE_ONLY" == "1" ]]; then
  PY_ARGS+=(--cache_only)
fi

# Keep compatibility with the original EXTRA_ARGS convention.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  method/dyn3d_sga/evaluate_llm_judge_3d.py \
  "${PY_ARGS[@]}" \
  ${EXTRA_ARGS:-}

# Cache should always exist after a successful run.
test -s "$CACHE_PATH"

# In cache-only mode there is intentionally no final judge JSON.
if [[ "$CACHE_ONLY" != "1" ]]; then
  test -s "$OUTPUT_PATH"
fi
