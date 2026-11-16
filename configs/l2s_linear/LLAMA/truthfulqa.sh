#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/../common.sh"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-$DEFAULT_LAYER}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"

# Standard additive steering: h_new = h + ALPHA * L2S(h).
ALPHA="${ALPHA:-1.0}"
apply_l2s_linear_common
