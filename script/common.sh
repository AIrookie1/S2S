#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x python ]]; then PYTHON_BIN=python; else PYTHON_BIN=python; fi
fi
FEATURE_DIR="${FEATURE_DIR:-$REPO_ROOT/features}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$REPO_ROOT/prototypes}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results}"
mkdir -p "$FEATURE_DIR" "$CHECKPOINT_ROOT" "$PROTOTYPE_DIR" "$RESULT_ROOT"
export REPO_ROOT PYTHON_BIN FEATURE_DIR CHECKPOINT_ROOT PROTOTYPE_DIR RESULT_ROOT
