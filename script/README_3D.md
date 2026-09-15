# 3d-dyn adaptive 2-Slerp

The migrated pipeline computes `(alpha, beta, gamma)` per token from `X`
(normalized current hidden state), `V` (normalized learned mean), and `R`
(the L2-normalized router residual `coeff @ basis`). `R` is normalized in both
training and inference so the router learns residual direction. Pairwise similarity attention
uses `WEIGHT_KAPPA`; the three adaptive weights are positive and sum to one.

```bash
FOLD_IDX=0 WEIGHT_KAPPA=5.0 WEIGHT_TEMPERATURE=1.0 \
  ./script/dyn3d_sga/LLAMA/run_3d_eval.sh
```

Model-specific launchers are organized under `LLAMA/` and `QWEN/`. Each
folder contains the complete extract, train, evaluate, and end-to-end pipeline.

The implementation normalizes only the hidden-state copy consumed by the
Router and the mean direction. The Router residual stays as raw `coeff @ basis`
through training and inference. During evaluation tqdm reports `TokenSteer: xx.x%`
and the mean adaptive weights, followed by final token counts and mean weights.

## Sparse Top-K residual-router training

Router training uses sparse Top-K routing over the learned SVD basis. Only the
selected K basis coefficients are kept, and they are renormalized to sum to
one. The supervised target is `normalize((h_pos - h_neg) - v_avg)`, while the
prediction is the unnormalized `coeff @ basis`. The optimization objective is
`MSE(coeff @ basis, normalized_true_residual) + lambda * load_balance`. A
Switch-style load-balancing auxiliary loss discourages basis collapse. The
launchers expose Top-K and the load-balance weight as environment variables:

```bash
ROUTER_TOP_K=4 LOAD_BALANCE_LOSS_WEIGHT=0.01 \
  ./script/dyn3d_sga/LLAMA/run_3d_train.sh
```

The effective Top-K and load-balance weight are stored in each dynSVD
checkpoint, so evaluation automatically uses the same sparse routing behavior.
Legacy checkpoints without `router_top_k` remain dense for backward
compatibility.
