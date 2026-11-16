#!/usr/bin/env bash
VAL_FRACTIONS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
read -r -a SEED_VALUES <<< "${SEEDS:-42 123 0}"
FOLD_IDX="${FOLD_IDX:-0}"
