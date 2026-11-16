#!/usr/bin/env bash
MODEL="${LLAMA_MODEL:-llama3.1-8B-Instruct}"
MODEL_DIR="${LLAMA_MODEL_DIR:-models/Meta-Llama-3.1-8B-Instruct}"
MODEL_LABEL="${LLAMA_MODEL_LABEL:-LLAMA}"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-float16}"
