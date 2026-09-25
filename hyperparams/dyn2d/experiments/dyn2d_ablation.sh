#!/usr/bin/env bash
# Dyn2D TruthfulQA 6x6 ablation grid.
# Model/fold selection is controlled by experiments/dyn2d_ablation/run.sh:
#   MODELS="LLAMA QWEN" FOLDS="0 1"
DYN2D_ABLATION_STRENGTHS=(1.0 2.0 3.0 4.0 5.0 6.0)
DYN2D_ABLATION_SLERPS=(0.0 0.2 0.4 0.6 0.8 1.0)
