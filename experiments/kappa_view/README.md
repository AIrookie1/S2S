# Kappa view

This experiment visualizes the token-level Dyn3D adaptive weights
`(alpha, beta, gamma)` as text-free ternary density plots.

Defaults are intentionally matched to the current project:

- model: `LLAMA`
- fold: `0`
- kappa values: `1 3 5 10`
- checkpoint: the standard current Dyn3D checkpoint under
  `checkpoints/dyn3d_sga/truthfulqa/...`
- gate mode: `all` (all tokens that receive adaptive weights)

Run:

```bash
bash experiments/kappa_view/plot.sh
```

To visualize only tokens that actually pass the Dyn3D margin gate:

```bash
WEIGHTS_GATE_MODE=steered bash experiments/kappa_view/plot.sh
```

If the standard checkpoint is missing, opt in to training with `RUN_TRAIN=1`.
If features are also missing, add `RUN_EXTRACT=1`.

All plot size/style/text controls are centralized in:

- `experiments/plot_sizes.py`
- `experiments/plot_theme.py`
- `experiments/plot_specs.py`
