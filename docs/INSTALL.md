# Installation

Run commands from the repository root. Python 3.10 is the tested interpreter.

## Main environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/main.txt
python -m pytest -q
```

The dependency profile records the tested main environment. Full runs require a CUDA host and a compatible PyTorch installation; GPU experiments are not run by the lightweight CI workflow.

## Model setup

```bash
python tools/download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct --output models/Llama
python tools/download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct --output models/Qwen
```

The helper records the resolved model revision. Replay a fixed snapshot by passing `--revision` with its full commit SHA. Complete gated-model access locally; do not commit authentication tokens.

```bash
python -m reproducibility run table1 --model llama --model-dir /path/to/local/model --dry-run
python -m reproducibility run table1 --config configs/reproduction/qwen.json --execute --resume
```

## Scaling environment

```bash
python3.10 -m venv .venv-scaling
.venv-scaling/bin/python -m pip install -r requirements/scaling.txt
.venv-scaling/bin/python -m pytest -q tests/rebuttal_R1W2_model_scaling
.venv-scaling/bin/python -m reproducibility run r1-scales --execute
```

Main and scaling profiles intentionally use different Transformers versions. The 14B scaling condition is labeled NF4; the other declared scaling conditions use BF16.
