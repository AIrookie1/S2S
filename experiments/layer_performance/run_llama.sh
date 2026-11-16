#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"
# Both methods use the same direct-vector MLP training configuration.
set -a
source "$REPO_ROOT/hyperparams/dyn3d_sga/LLAMA/truthfulqa.sh"
set +a
# Keep orchestration in this shell: Python sources participate in cache hashes.
# These subsets of the original grid can reuse already completed evaluations.
SEARCH_PROFILE="${SEARCH_PROFILE:-fast}"
case "$SEARCH_PROFILE" in
  fast)
    strength_grid="1 2"
    slerp_grid="0.2 0.6"
    kappa_grid="3 10"
    margin_grid="-0.7 -0.3"
    ;;
  balanced)
    strength_grid="1 2 3"
    slerp_grid="0.2 0.6 1"
    kappa_grid="1 3 10"
    margin_grid="-0.7 -0.3 0.3"
    ;;
  full)
    strength_grid="0.5 1 1.5 2 2.5 3"
    slerp_grid="0 0.2 0.4 0.6 0.8 1"
    kappa_grid="1 2 3 5 10 20"
    margin_grid="-0.9 -0.7 -0.5 -0.3 0 0.3"
    ;;
  *)
    echo "Unknown SEARCH_PROFILE=$SEARCH_PROFILE; use fast, balanced or full" >&2
    exit 2
    ;;
esac
# Explicit environment values and run.py CLI arguments override the preset.
export STRENGTH_VALUES="${STRENGTH_VALUES:-$strength_grid}"
export SLERP_T_VALUES="${SLERP_T_VALUES:-$slerp_grid}"
export WEIGHT_KAPPA_VALUES="${WEIGHT_KAPPA_VALUES:-$kappa_grid}"
export MARGIN_VALUES="${MARGIN_VALUES:-$margin_grid}"
echo "[search] profile=$SEARCH_PROFILE (CLI/environment grid overrides take precedence)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/envs/py310/bin/python
  else
    PYTHON_BIN=python
  fi
fi
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
exec "$PYTHON_BIN" "$HERE/run.py" "$@"
