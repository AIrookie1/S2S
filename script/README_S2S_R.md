# S2S_R adaptive 2-Slerp

S2S_R keeps the original three-direction spherical intervention, but its dynamic
residual is now produced by a direct vector MLP instead of a basis router.

For each training pair:

```text
v_i = h_pos - h_neg
MLP(normalize(h_neg)) -> v_hat_i
loss = MSE(v_hat_i, v_i)
```

`positive_mode=mean` preserves the original definition of `h_pos`: all positive
answer activations for a question are averaged before forming `v_i`.

The selected outer fold is split at question level. The complementary
`1-VAL_FRACTION` subset estimates `v_avg`, while `VAL_FRACTION` trains the MLP.
Only negative hidden states are used as MLP inputs; positive states and
interpolated auxiliary samples are not added.

At inference time, the hidden-state copy consumed by the MLP is L2-normalized.
One vector is predicted from the last hidden state of the current intervention
segment:

```text
v_hat_i = MLP(normalize(h_last))
R = normalize(v_hat_i - v_avg)
```

That single residual direction is broadcast across the segment, exactly
preserving the previous sequence-level S2S_R behavior. The three unit directions
are then:

- `X`: normalized current token hidden state,
- `V`: normalized `v_avg`,
- `R`: normalized predicted residual.

The existing similarity-graph adaptive `(alpha, beta, gamma)` weighting,
symmetric two-Slerp composition, hidden-norm restoration, and margin gating are
unchanged.

```bash
FOLD_IDX=0 WEIGHT_KAPPA=5.0 \
  ./script/S2S_R/LLAMA/run_S2S_R.sh
```

Training hyperparameters now expose `MLP_HIDDEN`, `EPOCHS`, `BATCH_SIZE`, and
`LEARNING_RATE`. Basis rank/energy, sparse routing, Top-K, load balancing, and
auxiliary interpolation settings are no longer part of S2S_R.
