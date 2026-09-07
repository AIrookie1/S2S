#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/envs/py310/bin/python
  else
    PYTHON_BIN=python
  fi
fi

MODEL="${MODEL:-llama3.1-8B-Instruct}"
LAYER="${LAYER:-14}"
GPU="${GPU:-0}"
FEATURE_DIR="${FEATURE_DIR:-$REPO_ROOT/features}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$REPO_ROOT/prototypes}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results}"
JUDGE_DIR="${JUDGE_DIR:-$REPO_ROOT/results_llm_judge}"
MODEL_DIR="${MODEL_DIR:-}"

MODEL_DIR_ARGS=()
if [[ -n "$MODEL_DIR" ]]; then
  MODEL_DIR_ARGS+=(--model_dir "$MODEL_DIR")
fi

mkdir -p "$FEATURE_DIR" "$PROTOTYPE_DIR" "$RESULT_DIR" "$JUDGE_DIR"
echo "Spherical-Steering | repo=$REPO_ROOT | python=$PYTHON_BIN | model=$MODEL | layer=$LAYER | gpu=$GPU"
