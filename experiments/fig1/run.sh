#!/usr/bin/env bash
set -euo pipefail

# Keep the historical implementation in fig1_train_mean while exposing the
# conventional experiments/fig1/run.sh entry point.
exec "$(cd "$(dirname "$0")" && pwd)/../fig1_train_mean/run.sh" "$@"
