# Dyn3D ablation experiments

All ablations source `hyperparams/base.sh` through an experiment-specific config. The baseline is `energy=0.99`, `router_top_k=40`, `load_balance_loss_weight=0.02`, `weight_kappa=8`, and `margin=-0.3`. Each ablation changes only its named variable.

Directories:
- `load_balance_ablation`: load-balancing weight = 0.01 ... 0.09.
- `router_topk_ablation`: requested Top-K = 1, 5, 10, 20, 40, 80. The checkpoint and summary record both requested and effective Top-K if energy-selected rank clamps K.
- `val_fraction_ablation`: validation-fraction sweep with multiple seeds.
- `weight_kappa_ablation`: evaluation-only kappa sweep (includes baseline 8).
- `margin_ablation`: evaluation-only margin sweep (includes baseline -0.3).

Run `bash experiments/<name>/run.sh`. Add `DRY_RUN=1` to validate command construction without model/data execution. Add `RUN_EXTRACT=1` to generate missing TruthfulQA features using the shared extraction configuration.
