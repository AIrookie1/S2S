#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/extract_features.sh"
bash "$SCRIPT_DIR/compute_prototypes.sh"
bash "$SCRIPT_DIR/run_mc_all.sh"
bash "$SCRIPT_DIR/run_llm_judge_all.sh"
