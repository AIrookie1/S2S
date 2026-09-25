#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_CONFIG="$(cd "$CONFIG_DIR/.." && pwd)"
source "$MODEL_DIR_CONFIG/../common.sh"
source "$MODEL_DIR_CONFIG/model.sh"
DATASET="copa"
LAYER="${LAYER:-16}"
GPU="${GPU:-2}"
NUM_FOLDS="${NUM_FOLDS:-1}"
FOLD_IDX="${FOLD_IDX:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.9}"
MLP_HIDDEN="${MLP_HIDDEN:-1024}"
EPOCHS="${EPOCHS:-400}"
# Initial generic default; override at launch time with ALPHA=<value>.
ALPHA="${ALPHA:-1.4}"
apply_l2s_linear_common
