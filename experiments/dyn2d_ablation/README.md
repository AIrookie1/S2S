# Dyn2D strength × SLERP ablation

This experiment evaluates the Dyn2D steering grid on TruthfulQA for both supported models and both outer folds.

Default sweep:

- models: `LLAMA QWEN`
- folds: `0 1`
- strength: `0.5 1.0 1.5 2.0 2.5 3.0`
- SLERP coefficient: `0.0 0.2 0.4 0.6 0.8 1.0`

For every model, fold0 and fold1 are trained/evaluated independently. The final grid value is the arithmetic mean of the two fold metrics at the same `(strength, slerp_t)` coordinate. Llama and Qwen are aggregated separately.

Run:

```bash
bash experiments/dyn2d_ablation/run.sh
```

Useful overrides:

```bash
GPU=0 MODELS="LLAMA QWEN" FOLDS="0 1" RUN_EXTRACT=auto RUN_TRAIN=auto \
  bash experiments/dyn2d_ablation/run.sh
```

`LLAMA_GPU` and `QWEN_GPU` can override the global GPU separately. `LLAMA_FEATURE_PATH` and `QWEN_FEATURE_PATH` can override feature files separately.

Outputs are written under `results/experiment/dyn2d_ablation/`:

- `checkpoints/<model>/fold*/`: fold-specific Dyn2D MLP checkpoints
- `raw/<model>/fold*/`: raw fold-specific 6×6 CSV/JSON results
- `summaries/<model>_fold_mean.csv|json`: fold0/fold1 arithmetic means (MC1/MC2/MC3)
- `figures/`: one MC1 heatmap per model, using fold means

`REUSE_EXISTING=1` (default) resumes completed grid cells instead of recomputing them.
