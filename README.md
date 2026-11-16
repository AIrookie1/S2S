# S2S: anonymous reproduction package

Source code and explicit reproduction plans for S2S-L/S2S-R, the main baselines,
and the experiments discussed during review. The supported commands below keep
scientific implementations under `method/` intact and isolate new outputs.

## What is included

| Experiment | Entry point | Scope |
|---|---|---|
| Main TruthfulQA comparison / Q1 | `python -m reproducibility.run main --model llama` | Baseline, additive mean, L2S, Spherical, S2S-L, S2S-R; both folds |
| Generic main tasks | `python -m reproducibility.run generic --model llama` | BoolQ, COPA, StoryCloze, MMLU-Global, Winogrande; baseline plus four methods |
| Q2 residual controls | `python -m reproducibility.run component --model llama` | All six default ablation rows AND the review's exact mean-control setting, both folds |
| Q3 correct/incorrect gate rates | `python -m reproducibility.run gate --model llama` | Applied and observe-only candidate analysis |
| Q3 margin / closed gate | `python -m reproducibility.run margin --model llama` | Includes default -0.7 and closed-gate margin 1.0; MC and label-conditioned rates |
| Target noise / Q1 configuration comparison | `python -m reproducibility.run noise --model llama` | Both folds; includes clean sigma=0; target noise and inference noise distinguished |
| Q3 model scaling | `python -m reproducibility.run scaling` | Qwen2.5 0.5B/1.5B/3B BF16, 14B NF4; separate dev selection and per-fold outputs |
| BiPO baseline | `python -m reproducibility.run bipo --model llama --model-dir models/Llama` | Existing BiPO adaptation, BiPO alone and calibrated variants; two folds |
| Q3 free generation / text safety adapter | `python -m reproducibility.run generation --model llama --prompts data/mmsafe.jsonl` | Actual generated lengths/EOS; baseline/L2S/S2S-L/S2S-R; explicit safety scoring step |

Commands print a JSON plan by default. Add `--execute` to run models and
`--resume` to reuse matching, hash-verified outputs. `--model qwen` selects
Qwen2.5-7B-Instruct for the main/control suites. See [the experiment guide](docs/EXPERIMENTS.md).

## Install

Use Python 3.10 and a CUDA host for full model evaluations. CPU checks do not
require model downloads. The two Transformers environments are separate:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s reproducibility/tests -v
```

The main dependency versions were read from the existing experiment environment;
this is not a fresh-install validation on every CUDA platform. Install the
appropriate PyTorch CUDA build for your host. `baukit` is pinned to an upstream
commit. Scaling uses Transformers 5.x in a separate environment:

```bash
python -m venv .venv-scaling
.venv-scaling/bin/python -m pip install -r requirements-scaling.txt
.venv-scaling/bin/python -m reproducibility.run scaling
```

The anonymous export bundles the local TruthfulQA CSV and the small evaluation
library, with the upstream license, so ignored local dependencies are not lost.
Other benchmark datasets are obtained by their existing loaders; access-controlled
datasets such as StoryCloze require the provider's access procedure.

Download model weights separately. Complete the provider's access procedure for
gated models locally. The download helper resolves and records a full revision;
pass `--revision` to replay that exact snapshot later:

```bash
python tools/download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct --output models/Llama
python tools/download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct --output models/Qwen
python -m reproducibility.run main --model llama --model-dir models/Llama
python -m reproducibility.run main --model llama --model-dir models/Llama --execute --resume
python -m reproducibility.run main --model qwen --model-dir models/Qwen --execute --resume
```

New model snapshots are identified by the download receipt; a historical main-model
revision was not recorded in the original artifacts, so bit-for-bit equivalence to
an unspecified historical snapshot is not claimed.

## Read the existing evidence without a GPU

The generated anonymous release includes `artifacts/` when built with
`--include-results`:

- `REPORT.md`: archived main per-fold results and unresolved reviewer numbers.
- `main_per_fold.csv`, `main_averages.csv`: aligned fold labels, macro and pooled MC1/2/3.
- `evidence.json`: source-file hashes, recorded configurations, compact question scores.
- `number_provenance.json`: a lookup for each conflicting reviewer number.
- `audit_status.json`: validation errors and claims with no matching audited artifact.

See [fold definitions and manuscript wording](docs/FOLD_PROTOCOL.md) and
[known result discrepancies](docs/RESULT_PROVENANCE.md). Existing artifacts are
not a new execution of the experiments. Missing or conflicting records are not
filled with invented numbers.

## Repository layout

```text
method/                 scientific implementations
hyperparams/            existing per-model/per-task settings
script/                 existing low-level launchers
experiments/            individual studies, analyses, and tests
reproducibility/        unified plans, generation/safety adapter, audits, export
reproducibility/tests/  lightweight checks and optional CPU hook checks
tools/                  model download and anonymous publishing helpers
docs/                   protocols, reproducibility status, publication steps
artifacts/              compact archived evidence in the generated release
```

The unified entry points are the maintained reproduction path. Older exploratory
scripts are retained for provenance; some reference removed legacy modules and
are not covered by the supported commands. They are listed in the experiment guide.

## Build and publish an anonymous copy

From the working repository:

```bash
python -m reproducibility.build_release --output anonymous_release --include-results
python -m reproducibility.verify_release anonymous_release
```

Upload **only the generated `anonymous_release/` tree**. It contains no old Git
history, local checkpoints, binaries, editor settings, credentials, or machine
logs. Personal absolute paths are replaced in the export, and the export is
blocked if the content scan finds an identity or credential pattern. Required
third-party license attributions remain intact.

Follow [the Chinese publication walkthrough](docs/ANONYMOUS_PUBLICATION_ZH.md).
The scanner covers file content; an anonymous account or review proxy is still
needed to avoid exposing the owner of the original GitHub account. GitHub hosts
the source and CPU checks; full experiments run on a reviewer-controlled CUDA host.
