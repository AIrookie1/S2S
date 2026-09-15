#!/usr/bin/env bash
set -euo pipefail
EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENTS_DIR/.." && pwd)"
source "$REPO_ROOT/script/common.sh"
DRY_RUN="${DRY_RUN:-0}"; RUN_EXTRACT="${RUN_EXTRACT:-0}"
configure_model(){
  local key="${1^^}"
  unset MODEL MODEL_DIR MODEL_LABEL LAYER DEFAULT_LAYER DEFAULT_FOLD_IDX
  case "$key" in LLAMA|QWEN) source "$REPO_ROOT/hyperparams/dyn3d_sga/$key/truthfulqa.sh" ;; *) echo "Unknown model: $1" >&2; return 2 ;; esac
}
run_cmd(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '
'; else "$@"; fi; }
require_file(){ [[ "$DRY_RUN" == 1 || -s "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }; }
safe_tag(){ local value="$1"; value="${value//-/m}"; value="${value//./p}"; printf '%s' "$value"; }
prepare_feature(){
  local feature_path="$1"
  if [[ -s "$feature_path" ]]; then return 0; fi
  if [[ "$RUN_EXTRACT" != 1 ]]; then [[ "$DRY_RUN" == 1 ]] && return 0; echo "Missing feature file: $feature_path (set RUN_EXTRACT=1)" >&2; return 1; fi
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/extract_features.py "$MODEL" --model_dir "$MODEL_DIR" --layer "$LAYER" --save_dir "$FEATURE_DIR" --output_name "$(basename "$feature_path")" --batch_size "$EXTRACT_BATCH_SIZE" --device "$DEVICE" --dtype "$DTYPE" --split_seed "$SEED"
  require_file "$feature_path"
}
train_dyn3d(){
  local feature_path="$1" checkpoint_path="$2" fold_idx="$3" val_fraction="$4" seed="$5" router_top_k="$6" load_balance_weight="$7"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" method/dyn3d_sga/train_dynSVD.py --feature_path "$feature_path" --positive_mode "$POSITIVE_MODE" --output_path "$checkpoint_path" --num_folds "$NUM_FOLDS" --fold_idx "$fold_idx" --val_fraction "$val_fraction" --energy "$SVD_ENERGY" --router_hidden "$ROUTER_HIDDEN" --router_top_k "$router_top_k" --load_balance_loss_weight "$load_balance_weight" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --device "$DEVICE" --seed "$seed" --use_data_mode "$USE_DATA_MODE" --aux_interp_count "$AUX_INTERP_COUNT"
  require_file "$checkpoint_path"
}
eval_dyn3d(){
  local checkpoint_path="$1" output_path="$2" kappa="$3" margin="$4" seed="$5"
  run_cmd env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" test.py --method-dir dyn3d_sga --entry-file evaluate_3d_dyn.py -- "$MODEL" --checkpoint_path "$checkpoint_path" --model_dir "$MODEL_DIR" --layer "$LAYER" --eval_split "$EVAL_SPLIT" --weight_kappa "$kappa" --margin "$margin" --device "$DEVICE" --dtype "$DTYPE" --seed "$seed" --output_path "$output_path" --save_details
  require_file "$output_path"
}
