#!/usr/bin/env bash
LINEAR_ALPHA="${LINEAR_ALPHA:-1.0}"
# This file is sourced after the model-specific config, so use that model's depth.
LAYERS_DEFAULT="${LAYERS_DEFAULT:-$(seq -s " " 0 $((NUM_HIDDEN_LAYERS - 1)))}"
