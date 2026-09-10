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

## Sparse Top-K router training

Router training uses sparse Top-K routing over the learned SVD basis. Only the
selected K basis coefficients are kept, and they are renormalized to sum to
one. A Switch-style load-balancing auxiliary loss discourages basis collapse.
The launchers expose both settings as environment variables:

```bash
ROUTER_TOP_K=4 LOAD_BALANCE_LOSS_WEIGHT=0.01 \
  ./script/dyn3d_sga/LLAMA/run_3d_train.sh
```

The effective Top-K and load-balance weight are stored in each dynSVD
checkpoint, so evaluation automatically uses the same sparse routing behavior.
Legacy checkpoints without `router_top_k` remain dense for backward
compatibility.
