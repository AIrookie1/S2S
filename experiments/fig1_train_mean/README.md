# Fig. 1: answer-pair vectors versus their shared mean

This experiment evaluates only TruthfulQA MC1 on the outer-fold training
questions. It compares the following two interventions on each false answer
candidate `j`, using the MC1 best truthful answer `i` as the positive answer:

- **Answer-pair vector:** use the answer pair's own raw vector
  `v_ij = h_i+ - h_j-`.
- **Shared mean:** replace every `v_ij` with the arithmetic mean of all raw
  answer-pair vectors from the same outer-fold training split.

There is no within-question averaging and no L2 normalization in either
condition. The MC1 correct answer is always scored with no intervention. Only
the canonical Hugging Face MC1 false choices receive the selected vector; both
correct and false answers are mapped directly to the feature NPZ choices rather
than reconstructed from the CSV answer strings. With the default `ALPHA=1.0`, the
answer-pair condition uses the exact additive construction
`h_j- + (h_i+ - h_j-)` at the selected layer.

Run the default two folds and draw the MC1 SVG and PNG:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/fig1_train_mean/run.sh
```

To run only fold 0:

```bash
FOLDS="0" CUDA_VISIBLE_DEVICES=0 bash experiments/fig1_train_mean/run.sh
```

Optional environment overrides are `PYTHON`, `MODEL_DIR`, `ALPHA`, and
`FOLDS`. Result JSON/CSV files are written to `results/` with the prefix
`pair_vs_mean_`; the figure is written to
`figures/fig1_pair_vs_mean_mc1.{svg,png}`. The new names intentionally do not
reuse results from the earlier per-question-normalized implementation.
