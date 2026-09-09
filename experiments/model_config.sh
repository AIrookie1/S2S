#!/usr/bin/env bash
# Shared model configuration for experiment sweeps.
#
# Defaults match the model names/layers/directories already used by the
# repository's dyn3d_sga LLAMA/QWEN scripts. Every value can be overridden
# through an environment variable.

configure_model() {
  local key="${1,,}"

  case "$key" in
    llama|llama3|llama3.1)
      MODEL="${LLAMA_MODEL:-llama3.1-8B-Instruct}"
      MODEL_LABEL="${LLAMA_MODEL_LABEL:-Llama 3.1 8B}"
      LAYER="${LLAMA_LAYER:-14}"
      EXP_MODEL_DIR="${LLAMA_MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct}"
      ;;
    qwen|qwen2.5)
      MODEL="${QWEN_MODEL:-Qwen2.5-7B-Instruct}"
      MODEL_LABEL="${QWEN_MODEL_LABEL:-Qwen 2.5 7B}"
      LAYER="${QWEN_LAYER:-19}"
      EXP_MODEL_DIR="${QWEN_MODEL_DIR:-/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct}"
      ;;
    *)
      echo "Unknown model key: $1 (expected: llama or qwen)" >&2
      return 2
      ;;
  esac

  # Explicitly reset MODEL_DIR for every model so common.sh's default Qwen
  # directory cannot leak into the Llama run.
  MODEL_DIR="$EXP_MODEL_DIR"

  export MODEL MODEL_LABEL LAYER EXP_MODEL_DIR MODEL_DIR
}

# Backward compatibility for older experiment scripts that source this file
# and immediately expect MODEL/LAYER/EXP_MODEL_DIR to exist.
configure_model "${EXP_MODEL_KEY:-qwen}"
