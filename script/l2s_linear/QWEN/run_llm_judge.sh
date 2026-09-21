#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/l2s_linear/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/l2s_linear/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_l2s_linear_fold${FOLD_IDX}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/l2s_linear/$MODEL/llm_judge}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
mkdir -p "$OUTPUT_DIR"
test -s "$CHECKPOINT_PATH"

PY_ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"
  --alpha "$ALPHA" --output_dir "$OUTPUT_DIR"
  --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$JUDGE_PRESET" --max_new_tokens "$MAX_NEW_TOKENS" --device "$DEVICE" --dtype "$DTYPE"
  --seed "$SEED")
[[ "${DISABLE_STEERING:-0}" == 1 ]] && PY_ARGS+=(--disable_steering)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/l2s_linear/evaluate_llm_judge.py "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
