# Dyn3D experiments

Dyn3D now uses a direct MLP vector predictor: `normalize(h_neg) -> v_i`, with
`v_i = h_pos - h_neg`. The dynamic residual used by 3D steering is
`normalize(v_hat_i - v_avg)`. There is no SVD basis or routing stage.

Current experiment groups:

- `margin_ablation`: varies the Dyn3D margin gate at evaluation time.
- `mlp_hidden_ablation`: varies Dyn3D direct-MLP hidden width over 16..2048 for Llama/Qwen and averages fold0/fold1.
- `dyn2d_ablation`: evaluates the Dyn2D strength × SLERP grid for Llama/Qwen and averages fold0/fold1 per grid cell.
- `weight_kappa_ablation`: varies adaptive similarity-weight sharpness.
- `val_fraction_ablation`: varies the within-fold fraction allocated to direct-MLP training; the complementary fraction estimates `v_avg`.
- `steering_vector_ablation`: compares pair-specific/shared-mean and linear/spherical steering using raw pair vectors.
- `component_ablation`: paper Table 3, five-row S2S-R component ablation (MLP / S2S / adaptive weighting / margin gate), fold 0 only.
- `generation_efficiency`: paper Table 4, one-GPU generation throughput with both generation-only and end-to-end timing for S2S-L/R.
- `truthfulqa`: TruthfulQA comparison utilities.
- `word_number`: plotting/analysis utilities retained from the project.
- `kappa_view`: token-level ternary density of Dyn3D adaptive weights for configurable kappa values and optional margin-gate filtering.

The obsolete basis-routing experiments were removed because the direct-vector
Dyn3D model has no basis coefficients, sparse Top-K selection, or routing-load
objective.

## Unified figure formatting

Experiment plotting is now centralized in three files:

- `plot_sizes.py`: four standard 1/2 × 1/4 width/height combinations plus special custom sizes.
- `plot_theme.py`: font/tick/legend/marker/line/errorbar sizes, colors and save settings.
- `plot_specs.py`: per-figure labels, visibility switches, legend/annotation positions and layouts.

See `PLOTTING.md` for the editing guide.
