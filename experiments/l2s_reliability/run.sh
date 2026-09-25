#!/usr/bin/env bash
set -euo pipefail

# Full evaluation + plotting:
#   CHECKPOINT_TEMPLATE='..._l2s_fold{fold}.npz' \
#   MODEL='llama3.1-8B-Instruct' \
#   bash experiments/l2s_reliability/run.sh
#
# Plot existing CSV results only:
#   bash experiments/l2s_reliability/run.sh --plot-only
#
# Plot only one existing figure:
#   bash experiments/l2s_reliability/run.sh --plot-only --figure performance
#   bash experiments/l2s_reliability/run.sh --plot-only --figure reliability

MODEL="${MODEL:-llama3.1-8B-Instruct}"
FEATURE_PATH="${FEATURE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/experiment/l2s_reliability}"
PLOT_ONLY=0
FIGURE="all"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/l2s_reliability/run.sh \
    [--plot-only] \
    [--figure all|performance|reliability] \
    [--output-dir DIR]

Modes:
  default      Run evaluation, then generate separate figures.
  --plot-only  Read existing CSV results and only plot.

Options:
  --figure     all (default), performance, or reliability
  --output-dir Result directory

Environment variables:
  MODEL
  CHECKPOINT_TEMPLATE   Required in evaluation mode and must contain {fold}
  FEATURE_PATH          Optional feature NPZ override
  OUTPUT_DIR

Examples:
  CHECKPOINT_TEMPLATE='checkpoints/l2s/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_l2s_fold{fold}.npz' \
  MODEL='llama3.1-8B-Instruct' \
  bash experiments/l2s_reliability/run.sh

  bash experiments/l2s_reliability/run.sh --plot-only

  bash experiments/l2s_reliability/run.sh \
    --plot-only \
    --figure performance

  bash experiments/l2s_reliability/run.sh \
    --plot-only \
    --figure reliability
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plot-only)
      PLOT_ONLY=1
      shift
      ;;
    --figure)
      if [[ $# -lt 2 ]]; then
        echo "Error: --figure requires all, performance, or reliability." >&2
        exit 2
      fi
      FIGURE="$2"
      if [[ "${FIGURE}" != "all" && "${FIGURE}" != "performance" && "${FIGURE}" != "reliability" ]]; then
        echo "Error: --figure must be all, performance, or reliability." >&2
        exit 2
      fi
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-dir requires a value." >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${PLOT_ONLY}" -eq 0 ]]; then
  CHECKPOINT_TEMPLATE="${CHECKPOINT_TEMPLATE:-}"
  if [[ -z "${CHECKPOINT_TEMPLATE}" ]]; then
    echo "Error: CHECKPOINT_TEMPLATE is required unless --plot-only is used." >&2
    exit 2
  fi
  if [[ "${CHECKPOINT_TEMPLATE}" != *"{fold}"* ]]; then
    echo "Error: CHECKPOINT_TEMPLATE must contain the literal placeholder {fold}." >&2
    echo "Example: ..._l2s_fold{fold}.npz" >&2
    exit 2
  fi

  EXTRA=()
  if [[ -n "${FEATURE_PATH}" ]]; then
    EXTRA+=(--feature_path "${FEATURE_PATH}")
  fi

  python experiments/l2s_reliability/evaluate.py "${MODEL}" \
    --checkpoint_template "${CHECKPOINT_TEMPLATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}"
else
  echo "Plot-only mode: using existing results from ${OUTPUT_DIR}"
fi

python experiments/l2s_reliability/plot.py \
  --input_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_DIR}/l2s_reliability" \
  --figure "${FIGURE}"
