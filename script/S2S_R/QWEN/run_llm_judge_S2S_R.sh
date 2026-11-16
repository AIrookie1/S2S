#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/S2S_R/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-2}"
CHECKPOINT_PATH_EXPLICIT=0
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH_EXPLICIT=1
else
  CHECKPOINT_PATH="$CHECKPOINT_ROOT/S2S_R/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_predictor_fold${FOLD_IDX}_${POSITIVE_MODE}.npz"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/S2S_R/$MODEL/llm_judge}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_S2S_R.json}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_base_probe_cache.pt}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-models/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-models/truthfulqa-info-judge-llama2-7B}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"; CACHE_ONLY="${CACHE_ONLY:-0}"

if [[ "$CHECKPOINT_PATH_EXPLICIT" == 0 && ! -s "$CHECKPOINT_PATH" ]]; then
  candidate="$REPO_ROOT/results/experiment/fig3c_gate_margin/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_baseline.npz"
  if [[ -s "$candidate" ]]; then
    CHECKPOINT_PATH="$candidate"
    echo "[judge-S2S_R] canonical checkpoint not found; using margin-ablation baseline $CHECKPOINT_PATH"
  fi
fi
mkdir -p "$OUTPUT_DIR"
if [[ ! -s "$CHECKPOINT_PATH" ]]; then
  echo "[judge-S2S_R] ERROR: checkpoint does not exist or is empty:" >&2
  echo "  $CHECKPOINT_PATH" >&2
  if [[ "$CHECKPOINT_PATH_EXPLICIT" == 0 ]]; then
    echo "[judge-S2S_R] Train it first with script/S2S_R/QWEN/run_S2S_R_train.sh, or set CHECKPOINT_PATH." >&2
  fi
  exit 1
fi
echo "[judge-S2S_R] model=$MODEL layer=$LAYER fold=$FOLD_IDX gpu=$GPU"
echo "[judge-S2S_R] checkpoint=$CHECKPOINT_PATH"
echo "[judge-S2S_R] output=$OUTPUT_PATH"

PY_ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"
  --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --cache_path "$CACHE_PATH"
  --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$JUDGE_PRESET" --max_new_tokens "$MAX_NEW_TOKENS"
  --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN")
[[ "$REBUILD_CACHE" == 1 ]] && PY_ARGS+=(--rebuild_cache)
[[ "$CACHE_ONLY" == 1 ]] && PY_ARGS+=(--cache_only)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/S2S_R/evaluate_llm_judge_S2S_R.py "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
test -s "$CACHE_PATH"; [[ "$CACHE_ONLY" == 1 ]] || test -s "$OUTPUT_PATH"
