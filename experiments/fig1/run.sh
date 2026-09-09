#!/usr/bin/env bash
set -euo pipefail

# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"

# Keep the historical implementation in fig1_train_mean while exposing the
# conventional experiments/fig1/run.sh entry point.
exec "$(cd "$(dirname "$0")" && pwd)/../fig1_train_mean/run.sh" "$@"
