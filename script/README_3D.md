# 3d-dyn adaptive 2-Slerp

The migrated pipeline now computes `(alpha, beta, gamma)` per token from the
three normalized directions `X` (current hidden state), `V` (learned mean
direction), and `R` (router residual direction). Pairwise similarity attention
uses `WEIGHT_KAPPA` (default `5.0`), and the final node-importance Softmax uses
`WEIGHT_TEMPERATURE` (default `0.5`). The three weights are positive and sum to
one automatically; they are no longer fixed evaluation hyperparameters.

```bash
FOLD_IDX=0 WEIGHT_KAPPA=5.0 WEIGHT_TEMPERATURE=1.0 \
  ./script/dyn3d_sga/LLAMA/run_3d_eval.sh
```

Model-specific launchers are organized under `LLAMA/` and `QWEN/`. Each
folder contains the complete extract, train, evaluate, and end-to-end pipeline.

The implementation normalizes the raw hidden state once, pre-normalizes the
mean direction once, and consumes the router output already normalized by
`normalize_basis_combination`; the adaptive-weight function itself does not
normalize again. During evaluation tqdm reports `TokenSteer: xx.x%` and the
mean adaptive weights, followed by final token counts and mean weights.
