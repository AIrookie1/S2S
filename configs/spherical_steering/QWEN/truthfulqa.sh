#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CONFIG_DIR/model.sh"
LAYER="${LAYER:-19}"
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.6}"
BETA="${BETA:-0.4}"
NUM_FOLDS="${NUM_FOLDS:-2}"
