#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/../common.sh"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-$DEFAULT_LAYER}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"
apply_dyn2d_common
