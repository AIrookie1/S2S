#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/hyperparams/dyn3d_sga/QWEN/truthfulqa.sh"
source "$REPO_ROOT/script/common.sh"

GPU="${GPU:-0}"
CHECKPOINT_PATH_EXPLICIT=0
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH_EXPLICIT=1
else
  CHECKPOINT_PATH="$CHECKPOINT_ROOT/dyn3d_sga/truthfulqa/$MODEL/${MODEL}_layer${LAYER}_dynSVD_fold${FOLD_IDX}_${POSITIVE_MODE}.npz"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/truthfulqa/dyn3d_sga/$MODEL/llm_judge}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_llm_judge_3d.json}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/${MODEL}_l${LAYER}_fold${FOLD_IDX}_base_probe_cache.pt}"
TRUTH_JUDGE_PATH="${TRUTH_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-truth-judge-llama2-7B}"
INFO_JUDGE_PATH="${INFO_JUDGE_PATH:-/mnt/nas-public-bing/pinxuanli11/truthfulqa-info-judge-llama2-7B}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"; CACHE_ONLY="${CACHE_ONLY:-0}"

if [[ "$CHECKPOINT_PATH_EXPLICIT" == 0 && ! -s "$CHECKPOINT_PATH" ]]; then
  for candidate in \
    "$REPO_ROOT/results/experiment/router_topk_ablation/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_topk${ROUTER_TOP_K}.npz" \
    "$REPO_ROOT/results/experiment/margin_ablation/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_baseline.npz" \
    "$REPO_ROOT/results/experiment/load_balance_ablation/checkpoints/$MODEL/fold${FOLD_IDX}/${MODEL}_l${LAYER}_fold${FOLD_IDX}_lb0p02.npz"; do
    if [[ -s "$candidate" ]]; then
      CHECKPOINT_PATH="$candidate"
      echo "[judge-3d] canonical checkpoint not found; using $CHECKPOINT_PATH"
      break
    fi
  done
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -s "$CHECKPOINT_PATH" ]]; then
  echo "[judge-3d] ERROR: checkpoint does not exist or is empty:" >&2
  echo "  $CHECKPOINT_PATH" >&2
  if [[ "$CHECKPOINT_PATH_EXPLICIT" == 0 ]]; then
    echo "[judge-3d] Train it first with script/dyn3d_sga/QWEN/run_3d_train.sh, or set CHECKPOINT_PATH." >&2
  fi
  exit 1
fi
echo "[judge-3d] model=$MODEL layer=$LAYER fold=$FOLD_IDX gpu=$GPU"
echo "[judge-3d] checkpoint=$CHECKPOINT_PATH"
echo "[judge-3d] output=$OUTPUT_PATH"

PY_ARGS=("$MODEL" --model_dir "$MODEL_DIR" --checkpoint_path "$CHECKPOINT_PATH" --layer "$LAYER"
  --output_dir "$OUTPUT_DIR" --output_path "$OUTPUT_PATH" --cache_path "$CACHE_PATH"
  --truth_judge_path "$TRUTH_JUDGE_PATH" --info_judge_path "$INFO_JUDGE_PATH"
  --preset "$JUDGE_PRESET" --max_new_tokens "$MAX_NEW_TOKENS"
  --weight_kappa "$WEIGHT_KAPPA" --margin "$MARGIN")
[[ "$REBUILD_CACHE" == 1 ]] && PY_ARGS+=(--rebuild_cache)
[[ "$CACHE_ONLY" == 1 ]] && PY_ARGS+=(--cache_only)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/evaluate_llm_judge_3d.py "${PY_ARGS[@]}" ${EXTRA_ARGS:-}
test -s "$CACHE_PATH"; [[ "$CACHE_ONLY" == 1 ]] || test -s "$OUTPUT_PATH"
