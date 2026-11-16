#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/s2s-model-ablation/venv/bin/python}"
MODEL_DIR="/mnt/nas-public-bing/pinxuanli11/s2s_ablation_models/Llama-3.1-Nemotron-Nano-8B-v1"
BASE_OUT="$ROOT/results/experiment/model_scaling_ablation/Llama-3.1-Nemotron-Nano-8B-v1"
while [[ ! -f "$MODEL_DIR/s2s_download.json" ]]; do sleep 30; done
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=Path('/mnt/nas-public-bing/pinxuanli11/s2s_ablation_models/Llama-3.1-Nemotron-Nano-8B-v1')
idx=json.loads((p/'model.safetensors.index.json').read_text())
missing=sorted(set(idx['weight_map'].values())-{x.name for x in p.glob('model-*.safetensors')})
if missing: raise SystemExit(f'missing shards: {missing}')
print('Nemotron shard validation passed')
PY
CUDA_VISIBLE_DEVICES=1,2 S2S_GPU_RESERVE_GIB=0 "$PYTHON_BIN" experiments/model_scaling_ablation/run.py --model nemotron_nano8b --model-dir "$MODEL_DIR" --dtype bfloat16 --cpu-memory 40GiB --output-dir "$BASE_OUT/bfloat16_fullgpu_e250" --epochs 250
CUDA_VISIBLE_DEVICES=1,2 S2S_GPU_RESERVE_GIB=0 "$PYTHON_BIN" experiments/model_scaling_ablation/shared_hparam_eval.py --source-dir "$BASE_OUT/bfloat16_fullgpu_e250" --output-dir "$BASE_OUT/shared_hparams_e250"
CUDA_VISIBLE_DEVICES=1,2 S2S_GPU_RESERVE_GIB=0 "$PYTHON_BIN" experiments/model_scaling_ablation/sweep_spherical_kappa.py --source-dir "$BASE_OUT/bfloat16_fullgpu_e250" --output-dir "$BASE_OUT/spherical_kappa_sweep"
