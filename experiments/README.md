# Dyn3D experiments

Dyn3D now uses a direct MLP vector predictor: `normalize(h_neg) -> v_i`, with
`v_i = h_pos - h_neg`. The dynamic residual used by 3D steering is
`normalize(v_hat_i - v_avg)`. There is no SVD basis or routing stage.

Current experiment groups:

- `margin_ablation`: varies the Dyn3D margin gate at evaluation time.
- `weight_kappa_ablation`: varies adaptive similarity-weight sharpness.
- `val_fraction_ablation`: varies the within-fold fraction allocated to direct-MLP training; the complementary fraction estimates `v_avg`.
- `steering_vector_ablation`: compares pair-specific/shared-mean and linear/spherical steering using raw pair vectors.
- `truthfulqa`: TruthfulQA comparison utilities.
- `word_number`: plotting/analysis utilities retained from the project.

The obsolete basis-routing experiments were removed because the direct-vector
Dyn3D model has no basis coefficients, sparse Top-K selection, or routing-load
objective.
