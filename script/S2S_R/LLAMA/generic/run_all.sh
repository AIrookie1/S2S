#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a DATASETS <<< "${DATASETS:-boolq copa storycloze mmlu_global winogrande}"
for dataset in "${DATASETS[@]}"; do bash "$SCRIPT_DIR/run_dataset.sh" "$dataset"; done

# GPU=0  PYTHONPA
# TH="$PWD:$PWD/method/S2S_R/generic${PYTHONPATH:+:$PYTHONPATH}" RUN_EXTRACT=1 RUN_TRAIN=0 
# RUN_EVAL=0 bash script/S2S_L/LLAMA/generic/run_dataset.sh mmlu_global