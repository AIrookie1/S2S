#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_CONFIG="$(cd "$CONFIG_DIR/.." && pwd)"
source "$MODEL_DIR_CONFIG/../common.sh"
source "$MODEL_DIR_CONFIG/model.sh"
DATASET="copa"
LAYER="${LAYER:-16}"
GPU="${GPU:-1}"
NUM_FOLDS="${NUM_FOLDS:-1}"
FOLD_IDX="${FOLD_IDX:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.5}"
MLP_HIDDEN="${MLP_HIDDEN:-256}"
EPOCHS="${EPOCHS:-2000}"
STRENGTH="${STRENGTH:-0.8}" #0.8
SLERP_T="${SLERP_T:-0.7}"
apply_S2S_L_common
