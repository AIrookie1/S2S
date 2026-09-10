#!/usr/bin/env bash
# Model-only settings. Shared training/evaluation hyperparameters live in base.sh.
HYPERPARAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HYPERPARAM_DIR/base.sh"

configure_model() {
  local model_key="${1:?model key required: llama or qwen}"
  case "$model_key" in
    llama)
      MODEL="${LLAMA_MODEL:-$HP_MODEL_LLAMA}"
      MODEL_DIR="${LLAMA_MODEL_DIR:-$HP_MODEL_DIR_LLAMA}"
      LAYER="${LLAMA_LAYER:-$HP_LAYER_LLAMA}"
      DEFAULT_FOLD_IDX="$HP_FOLD_LLAMA"
      MODEL_LABEL="Llama-3.1-8B-Instruct"
      ;;
    qwen)
      MODEL="${QWEN_MODEL:-$HP_MODEL_QWEN}"
      MODEL_DIR="${QWEN_MODEL_DIR:-$HP_MODEL_DIR_QWEN}"
      LAYER="${QWEN_LAYER:-$HP_LAYER_QWEN}"
      DEFAULT_FOLD_IDX="$HP_FOLD_QWEN"
      MODEL_LABEL="Qwen2.5-7B-Instruct"
      ;;
    *)
      echo "Unknown model key: $model_key (expected llama or qwen)" >&2
      return 2
      ;;
  esac
  MODEL_DIR_ARGS=()
  [[ -n "$MODEL_DIR" ]] && MODEL_DIR_ARGS+=(--model_dir "$MODEL_DIR")
}
