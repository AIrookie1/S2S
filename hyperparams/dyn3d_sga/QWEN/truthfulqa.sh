#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/../common.sh"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-$DEFAULT_LAYER}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"
VAL_FRACTION="${VAL_FRACTION:-0.5}"
WEIGHT_KAPPA="${WEIGHT_KAPPA:-3}"
MARGIN="${MARGIN:--0.7}"
apply_dyn3d_common
