# Steer the Steering: Calibrating Learned Interventions in Large Language Models

**Anonymous review release.** This repository contains the S2S_L/S2S_R implementations, paper experiments, and rebuttal studies. Archived scores and newly executed experiments are kept separate.

## Contents

- [Installation and model setup](docs/INSTALL.md)
- [Main experiments: Tables 1–4 and Figures 2–5](docs/EXPERIMENTS.md)
- [Rebuttal experiments](docs/REBUTTAL.md)
- [Complete experiment index](docs/EXPERIMENT_INDEX.csv)
- [Data and external dependencies](docs/DATA.md)

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/main.txt
python tools/download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct --output models/Llama
```

Scaling experiments use the separate [scaling environment](docs/INSTALL.md#scaling-environment). Model weights and gated benchmark access must be obtained from their providers.

## Quick start

```bash
python -m reproducibility list
python -m reproducibility run table1 --model llama --dry-run
python -m reproducibility run table1 --model llama --execute --resume
```

The CLI previews commands by default. Add `--execute` to run them; use `--model-dir` for an existing local model. Model presets are in [configs/reproduction](configs/reproduction), and all experiment routes are registered in [configs/experiments.json](configs/experiments.json).

## Repository structure

```text
methods/          S2S_L, S2S_R, and baseline implementations
configs/          Model presets, task hyperparameters, experiment registry
scripts/          Native extraction, training, and evaluation launchers
experiments/      Table/Figure-prefixed and rebuttal-prefixed studies
reproducibility/  Unified CLI, provenance, scoring, and anonymous export
requirements/     Separate main and scaling dependency profiles
tests/            Source, numerical, and integration checks
data/             Tracked TruthfulQA snapshot and local benchmark inputs
results/reference/ Archived scores and provenance; generated results stay local
reports/          Validation, source manifest, and privacy report
docs/             Installation and concise reproduction commands
third_party/      Preserved upstream license files
```

## Reproducibility

```bash
python -m pytest -q
python -m reproducibility verify . --allow-git
```

The test suite checks code/configuration behavior; it does not certify full GPU benchmark accuracy. Known missing inputs and protocol boundaries are recorded in [docs/DATA.md](docs/DATA.md) and the experiment index. In particular, inference residual noise and training-target noise are separate experiments.

## Anonymous export

```bash
python -m reproducibility export --output dist/s2s-anonymous
python -m reproducibility verify dist/s2s-anonymous --strict
```

Publish the exported copy. It excludes Git history, model weights, runtime outputs, and local environment files. See [anonymous publication notes](docs/ANONYMITY.md).

## Attribution

Third-party code and data retain their upstream notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Repository organization references are listed in [docs/REFERENCES.md](docs/REFERENCES.md). Author-identifying metadata is withheld during anonymous review.
