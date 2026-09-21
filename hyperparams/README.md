# Hyperparameter layout

Hyperparameters are grouped by steering method first, then by model.

- `dyn3d_sga/common.sh`: parameters genuinely shared by Dyn3D runs.
- `dyn3d_sga/LLAMA|QWEN/truthfulqa.sh`: model-specific TruthfulQA settings; Llama/Qwen keep their own model, layer and default fold.
- `dyn3d_sga/LLAMA|QWEN/generic/*.sh`: generic-task settings. Dataset hyperparameters are intentionally identical between Llama and Qwen; only model identity/path differs.
- `dyn3d_sga/experiments/*.sh`: model-independent experiment sweep grids.
- `spherical_steering/LLAMA|QWEN/truthfulqa.sh`: values copied from the uploaded Spherical-Steering quickstarts.
- `spherical_steering/LLAMA|QWEN/generic/*.sh`: values copied from the uploaded generic quickstart; the same dataset settings are used for both models for now.
- `l2s_linear/LLAMA|QWEN/truthfulqa.sh`: L2S training defaults plus the standard additive `ALPHA`.
- `l2s_slerp/LLAMA|QWEN/truthfulqa.sh`: L2S training defaults plus the vMF/SLERP `KAPPA/ALPHA/BETA` values.

Every shell variable remains environment-overridable.
