#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/truthfulqa/config.sh"
source "$REPO_ROOT/hyperparams/models.sh"
configure_model qwen
GPU="${GPU:-2}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"
source "$REPO_ROOT/script/common.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$FEATURE_DIR/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$JUDGE_DIR/qwen_truthfulqa}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_3d_fixed_router.json}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_base_probe_cache.pt}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
REBUILD_CACHE="${REBUILD_CACHE:-1}"; CACHE_ONLY="${CACHE_ONLY:-0}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
PY_ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"
  --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --cache_path "$CACHE_PATH"
  --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$JUDGE_PRESET" --max_new_tokens "$MAX_NEW_TOKENS"
  --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN")
[[ "$REBUILD_CACHE" == 1 ]] && PY_ARGS+=(--rebuild_cache)
[[ "$CACHE_ONLY" == 1 ]] && PY_ARGS+=(--cache_only)
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/evaluate_llm_judge_3d.py "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
test -s "$CACHE_PATH"
[[ "$CACHE_ONLY" == 1 ]] || test -s "$OUTPUT_PATH"
