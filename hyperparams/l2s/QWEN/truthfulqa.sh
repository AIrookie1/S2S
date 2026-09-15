#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/../common.sh"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-$DEFAULT_LAYER}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"

# Keep the Spherical-Steering TruthfulQA defaults unchanged.
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.6}"
BETA="${BETA:-0.4}"
apply_l2s_common
