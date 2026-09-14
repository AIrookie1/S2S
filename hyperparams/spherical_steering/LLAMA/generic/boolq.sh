#!/usr/bin/env bash
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_CONFIG="$(cd "$CONFIG_DIR/.." && pwd)"
source "$MODEL_DIR_CONFIG/model.sh"
DATASET="boolq"
LAYER="${LAYER:-12}"
KAPPA="${KAPPA:-20.0}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:--1.0}"
