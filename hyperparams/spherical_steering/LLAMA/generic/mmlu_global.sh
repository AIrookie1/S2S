#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_CONFIG="$(cd "$CONFIG_DIR/.." && pwd)"
source "$MODEL_DIR_CONFIG/model.sh"
DATASET="mmlu_global"
LAYER="${LAYER:-28}"
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.8}"
BETA="${BETA:--0.1}"
