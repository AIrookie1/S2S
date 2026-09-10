# Fig. 6 — margin-threshold ablation

Fig. 6 evaluates the gated 3D steering margin threshold on the TruthfulQA
multiple-choice test split for both configured models and both folds. The
checkpoint is trained once per model/fold, then reused for:

```text
margin = 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

The evaluator steers a token only when

```text
max(beta, gamma) > alpha + margin
```

The plot reports mean ± sample SD across fold0/fold1 using the shared
Nature-style model colors from `plot_common1.py`.

Run the complete sweep and plot with:

```bash
bash experiments/fig6/run_margin.sh
```

Useful overrides include `MODELS`, `FOLDS`, `GPU`, `WEIGHT_KAPPA`,
`AUTO_PLOT`, and the training hyperparameters used by the other figure
scripts.
