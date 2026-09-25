#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn2d/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"
GPU="${GPU:-2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/dyn2d/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynMLP_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/dyn2d/$MODEL/llm_judge}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_2d.json}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_base_probe_cache.pt}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"; CACHE_ONLY="${CACHE_ONLY:-0}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
PY_ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"
  --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --cache_path "$CACHE_PATH"
  --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$JUDGE_PRESET" --max_new_tokens "$MAX_NEW_TOKENS"
  --strength "$STRENGTH" --slerp_t "$SLERP_T")
[[ "$REBUILD_CACHE" == 1 ]] && PY_ARGS+=(--rebuild_cache)
[[ "$CACHE_ONLY" == 1 ]] && PY_ARGS+=(--cache_only)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn2d/evaluate_llm_judge_2d.py "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
test -s "$CACHE_PATH"; [[ "$CACHE_ONLY" == 1 ]] || test -s "$OUTPUT_PATH"
