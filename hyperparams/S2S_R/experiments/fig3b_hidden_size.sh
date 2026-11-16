#!/usr/bin/env bash
read -r -a MLP_HIDDEN_VALUES <<< "${HIDDEN_SIZES:-16 32 64 128 256 512 1024 2048}"
