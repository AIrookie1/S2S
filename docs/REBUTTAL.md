# Rebuttal experiments

These commands use the same entry point as the main experiments. The original reviewer/question labels and native programs are retained in [EXPERIMENT_INDEX.csv](EXPERIMENT_INDEX.csv).

## R1 W1: inference residual-direction noise

```bash
python -m reproducibility run r1-noise --model llama --folds 0 --execute --resume
# Optional both-fold repeat:
python -m reproducibility run r1-noise --model llama --folds 0 1 --execute --resume
```

This keeps training targets clean and perturbs the inference residual direction. It is separate from `fig5-noise`.

## R1 W2 / R2 W6: models and parameter scales

```bash
.venv-scaling/bin/python -m reproducibility run r1-scales --execute
.venv-scaling/bin/python -m reproducibility run r1-models --execute
.venv-scaling/bin/python -m reproducibility run r2-spherical-grid --execute
.venv-scaling/bin/python -m reproducibility run r2-fixed-t --execute
```

Run the source model comparisons first; the fixed-t comparison also reads Figure 3(a)'s 7B/8B grids. Additional-model defaults retain the declared 250-epoch runs.

## R2 W5: intervention layers

```bash
python -m reproducibility run r2-layers --search-profile fast --execute
python -m reproducibility run r2-layers --search-profile full --execute
```

These are exploratory test-grid maxima, not development-selected estimates.

## BiPO baseline and learned-anchor calibration

```bash
python -m reproducibility run r2-bipo --execute
python -m reproducibility run r2-bipo-analysis --execute
python -m reproducibility run r2-bipo-calibration --execute
```

Prepare Table 1 features first. BiPO alone and calibrated variants remain separate rows and protocols.

## R3 Q1–Q3: per-fold controls and gate behavior

```bash
python -m reproducibility run table1 --model llama --execute --resume
python -m reproducibility run r3-mean-control --execute --resume
python -m reproducibility run r3-gates --execute --resume
python -m reproducibility run r3-margin --execute --resume
```

The mean-only review control uses kappa=2.5/margin=0.965 on both folds; default ablations remain separate.

## MMsafe: generation and safety

```bash
python -m reproducibility run mmsafe-generate --prompts data/mmsafe.jsonl --execute --resume
python -m reproducibility run mmsafe-score --generations results/reproduction/generation/llama3.1-8B-Instruct/fold0/generations.jsonl --scores data/mmsafe_scores_fold0.jsonl --judge-config data/mmsafe_judge.json --output-dir results/mmsafe/fold0 --execute
```

Repeat scoring for fold 1. Original data/judge inputs are required; supported schemas are in [DATA.md](DATA.md).

## Supplementary analyses in the supplied configuration notes

```bash
python -m reproducibility run supp-transfer --execute
python -m reproducibility run supp-prediction-errors --execute
python -m reproducibility run supp-vector-metrics --execute
python -m reproducibility run supp-geometry --execute
```

Prepare Table 2 artifacts first; prediction-error diagnostics also require transfer checkpoints. Sampled geometry diagnostics are not a new numbered paper benchmark.
