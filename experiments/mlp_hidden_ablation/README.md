# Dyn3D MLP hidden-size ablation

This experiment varies only the hidden width of Dyn3D's direct-vector MLP and keeps the standard TruthfulQA Dyn3D settings fixed.

Default hidden widths:

```text
16 32 64 128 256 512 1024 2048
```

Default experiment matrix:

- models: Llama 3.1 8B and Qwen 2.5 7B
- folds: fold0 and fold1
- Llama layer: 14
- Qwen layer: 19
- `val_fraction=0.5`
- `weight_kappa=3`
- `margin=-0.7`
- `positive_mode=mean`
- `epochs=2000`
- `learning_rate=0.001`
- `batch_size=256`
- `seed=42`
- evaluation split: TruthfulQA test split

For each `(model, hidden size)`, MC1/MC2/MC3 are evaluated independently on fold0 and fold1, then averaged arithmetically. The plot uses MC1 mean ± sample SD across the two folds.

Run:

```bash
bash experiments/mlp_hidden_ablation/run.sh
```

Optional overrides:

```bash
GPU=0 MODELS="LLAMA QWEN" FOLDS="0 1" RUN_EXTRACT=1 \
  bash experiments/mlp_hidden_ablation/run.sh
```

`LLAMA_GPU` and `QWEN_GPU` can select different GPUs. Existing checkpoints/evaluations are reused by default (`REUSE_EXISTING=1`).

Outputs are under `results/experiment/mlp_hidden_ablation/`:

- `checkpoints/<model>/fold*/`: one Dyn3D checkpoint per hidden width
- `evaluations/<model>/fold*/`: full TruthfulQA evaluation JSON per hidden width
- `summaries/<model>_fold*.json`: per-fold compact summaries
- `mlp_hidden_ablation_metrics.csv|json`: fold0/fold1 values and arithmetic means for MC1/MC2/MC3
- `mlp_hidden_ablation_mc1.{png,pdf,svg}`: MC1 hidden-width curve
- `mlp_hidden_ablation_mc1_data.csv`: fold values, mean and SD used by the figure

Cache safety: before reusing a checkpoint, `run.sh` validates the checkpoint's
stored `mlp_hidden`, fold, training hyperparameters, and seed. A mismatched or
old checkpoint is retrained automatically, and its evaluation is regenerated.
This prevents a file named, for example, `hidden16.npz` from silently reusing a
checkpoint that was actually trained with the default hidden width 256.
