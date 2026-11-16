# Dyn3D experiments

Dyn3D now uses a direct MLP vector predictor: `normalize(h_neg) -> v_i`, with
`v_i = h_pos - h_neg`. The dynamic residual used by 3D steering is
`normalize(v_hat_i - v_avg)`. There is no SVD basis or routing stage.

Current experiment groups:

- `geometry_decision_alignment`: sampled cross-benchmark mechanism study linking MLP geometry, S2S-R spherical fusion, adaptive weights, and answer-margin changes, with finite-difference checks and oracle diagnostics.

- `mlp_prediction_error`: direct-MLP training versus official validation/test angular and norm errors on five generic datasets, same-layer TruthfulQA two-fold references, and auxiliary train-split cross-validation.

- `gate_correctness`: Llama S2S-R correct/incorrect TruthfulQA candidate gate rates, paired two-fold held-out analysis, applied and observe-only modes on GPU 1.

- `model_scaling_ablation`: Qwen2.5-Instruct 0.5B/1.5B/3B/14B规模对照，并保留 Qwen3.5-9B / Qwen2.5-32B-Instruct；五方法 TruthfulQA 比较、模型专属特征与预测器、独立 dev 选择和两折 held-out 指标。24GB GPU 推荐 1.5B/3B BF16，14B NF4。

- `margin_ablation`: varies the Dyn3D margin gate at evaluation time.
- `mlp_hidden_ablation`: varies Dyn3D direct-MLP hidden width over 16..2048 for Llama/Qwen and averages fold0/fold1.
- `dyn2d_ablation`: evaluates the Dyn2D strength × SLERP grid for Llama/Qwen and averages fold0/fold1 per grid cell.
- `weight_kappa_ablation`: varies adaptive similarity-weight sharpness.
- `val_fraction_ablation`: varies the within-fold fraction allocated to direct-MLP training; the complementary fraction estimates `v_avg`.
- `steering_vector_ablation`: compares pair-specific/shared-mean and linear/spherical steering using raw pair vectors.
- `component_ablation`: paper Table 3, five-row S2S-R component ablation (MLP / S2S / adaptive weighting / margin gate), fold 0 only.
- `truthfulqa_generic_transfer`: Llama TruthfulQA-trained MLPs at each generic task layer, TruthfulQA fold-specific `v_avg`, and MC1 averaged over source folds 0/1.
- `layer_performance`: Llama blocks 0–31, per-layer S2S-L/R hyperparameter grids, two-fold MC means and test-set-sweep best parameters. Run `bash experiments/layer_performance/run_llama.sh`.
- `generation_efficiency`: paper Table 4, one-GPU generation throughput with both generation-only and end-to-end timing for S2S-L/R.
- `truthfulqa`: TruthfulQA comparison utilities.
- `word_number`: plotting/analysis utilities retained from the project.
- `kappa_view`: token-level ternary density of Dyn3D adaptive weights for configurable kappa values and optional margin-gate filtering.
- `residual_analysis`: three-panel residual study runnable with either S2S-R or S2S-L-Static: oracle alignment/shuffle controls, method-faithful residual-swap MC intervention, method-family Mean-only baseline (cross-fitted Spherical Steering for S2S-R; linear additive v_avg for S2S-L-Static), and residual-quality versus angular gain.

The obsolete basis-routing experiments were removed because the direct-vector
Dyn3D model has no basis coefficients, sparse Top-K selection, or routing-load
objective.

## Unified figure formatting

Experiment plotting is now centralized in three files:

- `plot_sizes.py`: four standard 1/2 × 1/4 width/height combinations plus special custom sizes.
- `plot_theme.py`: font/tick/legend/marker/line/errorbar sizes, colors and save settings.
- `plot_specs.py`: per-figure labels, visibility switches, legend/annotation positions and layouts.

See `PLOTTING.md` for the editing guide.
