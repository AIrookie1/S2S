# MLP prediction geometry: generic versus TruthfulQA

## Exact `l2s_reliability` diagnostics

After the official experiment has produced its checkpoint/feature manifest:

```bash
/root/miniconda3/envs/py310/bin/python experiments/mlp_prediction_error/run_reliability.py
```

Writes `results/experiment/mlp_prediction_error/official/reliability/` with
`report.md`, `summary.csv/json`, `fold_metrics.csv`, and
`diagnostics_question.csv`. This is the comparison to use when matching
`experiments/l2s_reliability`:

- Directly imports its `rowwise_vector_metrics` and `aggregate_by_question`.
- Uses the same `load_l2s_pairs` raw-pair loader, including stored TruthfulQA
  positive × negative pairs rather than averaging positive answers first.
- Reports **cosine similarity** (higher is better), not degrees, and
  **relative norm error** (dimensionless, lower is better).
- Averages pairs within each question, then pools question rows across folds
  like the reference plot's centroid (rather than averaging folds equally).
- Retains the current direct-MLP checkpoint and its `normalize(h_neg)` input
  rule. These models were trained on mean-positive targets; the new raw-pair
  diagnostics do not retrain them or replace them with L2S checkpoints.

The earlier angle/MAE outputs below remain a separate diagnostic protocol.

## Main experiment: official validation/test

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /root/miniconda3/envs/py310/bin/python experiments/mlp_prediction_error/run_official.py
```

Extracts the full labeled official BoolQ/COPA/Winogrande `validation` and
XStoryCloze `eval` splits. For MMLU, uses the repository's balanced official
`test` subset: shuffle each category with seed 42, train rows 0:500 versus
evaluation rows 500:1000 (2000 evaluation questions total). This is a subset
of official test, not the complete MMLU test corpus.

BoolQ/COPA evaluation data are loaded explicitly: the legacy unified feature
loader ignores the requested split for these two datasets. The experiment
uses the released prompts, raw last-token block outputs, FP16 model inference,
and no extra BOS token. It extracts each dataset at its configured layer.
The forward pass stops after that block; subsequent blocks/logits are unused.

Reuses the existing BoolQ/MMLU generic MLPs. Missing COPA/StoryCloze/Winogrande
MLPs are trained under `official/checkpoints/` with production defaults
(`num_folds=1`, 50% of cached training questions fit MLP, 50% estimate mean,
width 256, 2000 epochs, batch 256, lr 0.001, seed 42). Official evaluation
questions never train the MLP or estimate its mean. Numeric question IDs are
local to each source split; equal integers across splits are different rows.

TruthfulQA uses the existing same-layer two-fold checkpoints described below,
plus released L14 checkpoints. Its official labeled split is `validation`,
so TruthfulQA test results are question-disjoint outer-fold tests within that
split. The generic training fractions and sample sizes differ from TruthfulQA;
interpret dataset comparisons with that distinction.

Main outputs: `results/experiment/mlp_prediction_error/official/`
(`report.md`, `summary.csv/json`, `fold_metrics.csv`, `question_metrics.csv`,
`features/`, and missing-model `checkpoints/`). Feature NPZs record the official
source split, model path, layer, dtype, seed, and prompt definition.
`--skip-extract` reuses extracted features. Production checkpoints are untouched.

Before the main comparison, six rows per cached training feature file are
recomputed with the current model, prompts and tokenizer. This audit requires
cosine similarity above 0.995 and relative L2 difference below 2% for every
sampled row, allowing small FP16 batching differences. Its per-dataset values
are saved to `training_feature_audit.json`; this is a sampled protocol check.

## Auxiliary experiment: cached train-split cross-validation

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/py310/bin/python experiments/mlp_prediction_error/run.py
```

Uses Llama-3.1-8B-Instruct's cached generic training features for BoolQ (L12),
COPA (L16), StoryCloze (L27), MMLU Global (L28), and Winogrande (L16).
Generic MLPs are fitted in this experiment's own checkpoint directory with the
released training implementation: hidden width 256, GELU, 2000 epochs, Adam,
learning rate 0.001, batch size 256, seed 42, clean raw-vector targets.
Existing production checkpoints are not overwritten.

Two outer folds split sorted question IDs, matching the TruthfulQA trainer.
In each fold, half the questions are held out; the other half is split 50:50
between mean-vector estimation and MLP fitting using seed 42. Thus approximately
25% of all questions fit each MLP, 25% estimate its mean, and 50% form its test
set. This differs from the generic production default `NUM_FOLDS=1`, whose
checkpoints have no outer test questions. The test here is a **held-out subset
of the cached training split**, not the dataset's official validation/test set.
The script rejects overlapping question groups and verifies recomputed training
MSE against the saved training MSE.

TruthfulQA uses existing two-fold source MLPs from
`results/experiment/truthfulqa_generic_transfer/<model>/<dataset>/checkpoints/truthfulqa_fold{0,1}.npz`
at the same four layers, plus the released L14 two-fold MLPs. All use
`positive_mode=mean`: `v = mean(h_pos for this question) - h_neg`.
Dataset sizes and domains differ, so these are descriptive comparisons; raw
norm errors also depend on layer scale. Use same-layer angle and relative norm
error for comparisons.

Metrics compare `v_hat = MLP(normalize(h_neg))` to `v`:

- `angle_deg`: `acos(dot(v_hat,v)/(||v_hat|| ||v||))`, in degrees.
- `norm_abs_error`: `abs(||v_hat|| - ||v||)` in raw activation units.
- `norm_rel_error_pct`: `100 * abs(||v_hat|| - ||v||) / ||v||`.
- `norm_signed_error`: `||v_hat|| - ||v||`; negative means underestimation.
- `vector_l2_error`: `||v_hat-v||`, which also depends on direction.
- `residual_angle_deg`: angle between `v_hat-v_avg` and `v-v_avg`.

Angles involving zero vectors and relative errors with zero target norm are
undefined and excluded, with valid-pair counts in the fold output.
First average negative-answer pairs within each question, then questions
within each fold, then the two folds equally. Training MSE reproduction instead
uses pair weighting, as the trainer does. There is no test-based selection or
early stopping.

Auxiliary outputs in `results/experiment/mlp_prediction_error/`:

- `checkpoints/`: experiment-specific generic MLPs and full training logs.
- `fold_metrics.csv`: metrics, valid counts, feature/checkpoint provenance.
- `question_metrics.csv`: per-question metrics for auditing.
- `summary.csv`, `summary.json`, `report.md`: two-fold comparisons.

Re-running reuses these generic checkpoints. `--epochs`, `--datasets`,
`--output`, and `--device` are configurable; checkpoint names include epochs.
TruthfulQA references remain the saved 2000-epoch models if `--epochs` changes.

Verify metric identities with:

```bash
/root/miniconda3/envs/py310/bin/python -m pytest experiments/mlp_prediction_error/test_metrics.py -q
```
