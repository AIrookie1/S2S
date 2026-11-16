#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_CONFIG="$(cd "$CONFIG_DIR/.." && pwd)"
source "$MODEL_DIR_CONFIG/../common.sh"
source "$MODEL_DIR_CONFIG/model.sh"
DATASET="storycloze"
LAYER="${LAYER:-27}"
GPU="${GPU:-1}"
NUM_FOLDS="${NUM_FOLDS:-1}"
FOLD_IDX="${FOLD_IDX:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.5}"
MLP_HIDDEN="${MLP_HIDDEN:-256}"
EPOCHS="${EPOCHS:-2000}"
# Initial generic default; override at launch time with ALPHA=<value>.
ALPHA="${ALPHA:-1.0}"
apply_l2s_linear_common
