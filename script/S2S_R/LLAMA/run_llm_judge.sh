#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_R/LLAMA/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/S2S_R/$MODEL/llm_judge}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-models/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-models/truthfulqa-info-judge-llama2-7B}"
mkdir -p "$OUTPUT_DIR"; test -s "$CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/S2S_R/evaluate_llm_judge.py "$MODEL"   --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"   --output_dir "$OUTPUT_DIR" --truth_judge_path "$TRUTH_JUDGE_PATH"   --info_judge_path "$INFO_JUDGE_PATH" ${EXTRA_ARGS:-}
