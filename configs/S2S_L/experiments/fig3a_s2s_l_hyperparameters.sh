#!/usr/bin/env bash
# S2S_L TruthfulQA full-dataset fine search grid.
# Model/fold selection is controlled by experiments/fig3a_s2s_l_hyperparameters/fig3a_run.sh:
#   MODELS="LLAMA QWEN" FOLDS="0 1"
S2S_L_ABLATION_STRENGTHS=(1.0 2.0 3.0 4.0 5.0)
S2S_L_ABLATION_SLERPS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
