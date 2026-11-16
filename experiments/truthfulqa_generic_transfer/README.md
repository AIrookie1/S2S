# TruthfulQA S2S-R transfer to generic tasks

Train Llama S2S-R MLPs on TruthfulQA at each target generic layer, retain the
TruthfulQA `v_avg` from each source checkpoint, and evaluate the generic task
using the existing S2S-R evaluator. Run both TruthfulQA outer folds (0 and 1);
report their arithmetic mean MC1 and sample SD. Each fold uses its own
TruthfulQA MLP and mean vector. Generic data is used only for evaluation.

## Configuration and data

Every dataset sources `hyperparams/dyn3d_sga/LLAMA/generic/<dataset>.sh` in its
own shell. These files control layer, MLP training settings, `VAL_FRACTION`,
seed, dtype, `weight_kappa`, and `margin`. TruthfulQA training always uses
`NUM_FOLDS=2`, folds 0 and 1; generic fold settings are not used.

Current defaults (read from the configs at run time):

| Dataset | Layer | Weight kappa | Margin |
|---|---:|---:|---:|
| BoolQ | 12 | 7 | 0.9 |
| COPA | 16 | 10 | 0 |
| StoryCloze | 27 | 2 | -0.5 |
| MMLU Global | 28 | 3 | 0.3 |
| Winogrande | 16 | 3 | -0.8 |

Use the existing TruthfulQA trainer and its question-level outer-fold/within-fold
split. Within each selected TruthfulQA fold, `VAL_FRACTION` of questions trains
the MLP; the complementary questions estimate `v_avg`. Positive answers are
averaged per question. The MLP target is raw `h_pos - h_neg`, and `v_avg` is the
mean of those raw vectors over the mean subset's negative-answer pairs.
Both the MLP weights and `v_avg` are copied unchanged for generic evaluation.
The two source folds may have different mean vectors.

Generic training features are neither extracted nor required. Dataset evaluation
splits, prompts, length-normalized option scoring, gating, and adaptive spherical
steering use the existing generic evaluator unchanged.

## Run

```bash
bash experiments/truthfulqa_generic_transfer/run.sh

# Inspect all extraction/training/preparation/evaluation commands.
DRY_RUN=1 bash experiments/truthfulqa_generic_transfer/run.sh

# Run one target on a chosen GPU.
GPU=0 DATASETS="boolq" bash experiments/truthfulqa_generic_transfer/run.sh

# Reuse existing TruthfulQA training checkpoints and evaluate with their means.
RUN_EXTRACT=0 RUN_TRAIN=0 bash experiments/truthfulqa_generic_transfer/run.sh
```

`RUN_EXTRACT=auto` reuses existing TruthfulQA features and extracts missing ones.
`RUN_EXTRACT=1` forces extraction; `RUN_EXTRACT=0` requires existing TruthfulQA
features. `RUN_TRAIN=1` retrains both source folds; `RUN_EVAL=1` evaluates and
summarizes. Use `RUN_EVAL=0` for preparation only. Standard model/path/GPU and
hyperparameter environment overrides work as in existing scripts.
`EXPERIMENT_RESULT_DIR` overrides the experiment output root.

Outputs default to
`results/experiment/truthfulqa_generic_transfer/<model>/<dataset>/`:

- `checkpoints/truthfulqa_fold{0,1}.npz`: source training checkpoints.
- `checkpoints/transfer_fold{0,1}.npz`: original MLP and TruthfulQA mean, with transfer provenance.
- `evaluations/truthfulqa_mean_fold{0,1}.json`: generic MC1 and steering statistics.
- `summary.json`, `summary.csv`: fold scores, mean, sample SD, and provenance.

The experiment root also contains `summary.json` and `summary.csv` for the
requested datasets. Scores are fractions in [0, 1]. Transfer checkpoints retain
source training/split/metrics fields and record model/layer, target dataset,
and TruthfulQA sources for both the MLP and `v_avg`. The collector checks that
MLP weights and mean match the original source checkpoint for each fold.

Existing source TruthfulQA checkpoints can be reused without retraining.
Previous generic-mean hybrid checkpoints and evaluation files are left in place;
new evaluations use the `truthfulqa_mean_fold*` filenames, and rerunning updates
the summaries to use those results.
