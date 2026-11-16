# Hyperparameter layout

Hyperparameters are grouped by steering method first, then by model.

- `S2S_R/common.sh`: parameters genuinely shared by S2S_R runs.
- `S2S_R/LLAMA|QWEN/truthfulqa.sh`: model-specific TruthfulQA settings; Llama/Qwen keep their own model, layer and default fold.
- `S2S_R/LLAMA|QWEN/generic/*.sh`: generic-task settings. Dataset hyperparameters are intentionally identical between Llama and Qwen; only model identity/path differs.
- `S2S_R/experiments/*.sh`: model-independent experiment sweep grids.
- `spherical_steering/LLAMA|QWEN/truthfulqa.sh`: values copied from the uploaded Spherical-Steering quickstarts.
- `spherical_steering/LLAMA|QWEN/generic/*.sh`: values copied from the uploaded generic quickstart; the same dataset settings are used for both models for now.
- `l2s_linear/LLAMA|QWEN/truthfulqa.sh`: L2S training defaults plus the standard additive `ALPHA`.
- `l2s_slerp/LLAMA|QWEN/truthfulqa.sh`: L2S training defaults plus the vMF/SLERP `KAPPA/ALPHA/BETA` values.

Every shell variable remains environment-overridable.
