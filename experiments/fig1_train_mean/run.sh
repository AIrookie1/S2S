#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-/root/miniconda3/envs/py310/bin/python}
PLOT_PYTHON=${PLOT_PYTHON:-$PYTHON}
MODEL_DIR=${MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct}
GPU=${CUDA_VISIBLE_DEVICES:-0}
ALPHA=${ALPHA:-1.0}
FOLDS=${FOLDS:-"0 1"}
export PYTHONDONTWRITEBYTECODE=1

if ! "$PLOT_PYTHON" -c "import matplotlib, numpy" >/dev/null 2>&1; then
  echo "Plotting requires matplotlib and numpy in PLOT_PYTHON=$PLOT_PYTHON" >&2
  echo "Install them first or set PLOT_PYTHON to a compatible Python interpreter." >&2
  exit 2
fi

RESULT_DIR=./experiments/fig1_train_mean/results
FIGURE_PREFIX=./experiments/fig1_train_mean/figures/fig1_pair_vs_mean_mc1
mkdir -p "$RESULT_DIR" "$(dirname "$FIGURE_PREFIX")"

result_files=()
for fold_idx in $FOLDS; do
  feature_path="./features/llama3.1-8B-Instruct_layer14_dynSVD_pairs_fold${fold_idx}.npz"
  output_path="${RESULT_DIR}/pair_vs_mean_fold${fold_idx}_alpha${ALPHA}.json"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiments/fig1_train_mean/evaluate_train_vectors.py \
    llama3.1-8B-Instruct \
    --feature_path "$feature_path" \
    --model_dir "$MODEL_DIR" \
    --layer 14 \
    --alpha "$ALPHA" \
    --output_path "$output_path" \
    --save_details
  result_files+=("$output_path")
done

"$PLOT_PYTHON" experiments/fig1_train_mean/plot_fig1.py \
  "${result_files[@]}" \
  --output_prefix "$FIGURE_PREFIX"
