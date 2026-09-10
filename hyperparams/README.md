# Hyperparameter layout

`base.sh` is the single canonical source for shared Dyn3D defaults. The current baseline uses SVD energy `0.99`, router Top-K `40`, load-balancing loss weight `0.02`, evaluation/judge `weight_kappa=8`, and `margin=-0.3`.

- `truthfulqa/config.sh`: shared main-pipeline settings used by LLAMA/QWEN scripts.
- `generic/<dataset>/config.sh`: five dataset-specific generic-task configurations.
- `experiments/<ablation>/config.sh`: one configuration per ablation. Only the ablated variable is swept; all other values inherit the baseline.

All shell variables remain environment-overridable, e.g. `GPU=1 EPOCHS=100 bash ...`.
