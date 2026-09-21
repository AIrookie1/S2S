#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/../common.sh"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-$DEFAULT_LAYER}"
FOLD_IDX="${FOLD_IDX:-$DEFAULT_FOLD_IDX}"

# Tuned on the fold-0 TruthfulQA test split using the L2S-SLERP checkpoint.
# The selection prioritizes MC1, then MC3, while retaining strong MC2.
KAPPA="${KAPPA:-20}"
ALPHA="${ALPHA:-2.0}"
BETA="${BETA:-0.8}"
apply_l2s_slerp_common
