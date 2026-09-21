# L2S-LINEAR

L2S-LINEAR learns the same raw vector predictor

```text
MLP(z_neg_raw) -> v_hat ~= v_T = z_pos_raw - z_neg_raw
```

Training uses ordinary MSE with raw hidden states and raw target vectors.
Inference is the standard additive steering rule, with no spherical
normalization and no vMF gate:

```text
h_new = h + alpha * v_hat
```

Only `alpha` controls the inference-time steering strength.
