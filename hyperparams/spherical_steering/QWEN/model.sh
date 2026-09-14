#!/usr/bin/env bash
MODEL="${QWEN_MODEL:-Qwen2.5-7B-Instruct}"
MODEL_DIR="${QWEN_MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct}"
MODEL_LABEL="${QWEN_MODEL_LABEL:-QWEN}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-float16}"
