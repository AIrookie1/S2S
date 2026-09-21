# Scripts

Scripts are grouped by method and model:

- `script/dyn3d_sga/LLAMA|QWEN/`
- `script/spherical_steering/LLAMA|QWEN/`
- `script/l2s_linear/LLAMA|QWEN/` (standard additive L2S)
- `script/l2s_slerp/LLAMA|QWEN/` (vMF-gated spherical L2S)

`dyn3d_sga` and `spherical_steering` also contain `generic/` launchers. The two L2S variants are TruthfulQA-only. Generic outputs are colocated by dataset under `results/generic/<dataset>/<method>/`.
