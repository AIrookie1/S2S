# Experiment changes

## Target-noise robustness
Added `experiments/l2s_noise_robustness/`.

Training corruption is applied only to the MLP target:
`v_tilde = v + sigma * ||v||_2 * normalize(epsilon)`, `epsilon ~ N(0,I)`.

Default sigma: `0, 0.1, 0.2, 0.4, 0.8`.
Held-out evaluation remains clean. The experiment compares TruthfulQA MC1 for:
L2S-Linear, L2S-SLERP, S2S_L_Static, and S2S_R.

The L2S and Dyn3D/S2S training APIs now expose `--target_noise_sigma` and
`--noise_seed`, and checkpoints record the noise definition/level/seed.

## Residual analysis
`experiments/residual_analysis/plot.py` now exports three independent figures:
`*_alignment`, `*_swap`, and `*_geometric_gain`, using the shared compact
publication theme and centralized single-figure size.

## L2S-LINEAR generic MC support
Added L2S-LINEAR training/evaluation launchers for BoolQ, COPA, StoryCloze,
MMLU-Global, and Winogrande for both LLAMA and QWEN.

The generic L2S path reuses the canonical generic answer-level features and
MC1 scorer, keeps the existing Dyn3D-compatible question split convention,
and applies standard additive inference `h_new = h + alpha * L2S(h)`.
Direct scripts live under `script/l2s_linear/<LLAMA|QWEN>/generic/`.
