# L2S-SLERP

L2S-SLERP learns the raw vector predictor

```text
MLP(z_neg_raw) -> v_hat ~= v_T = z_pos_raw - z_neg_raw
```

Training uses ordinary MSE with raw hidden states and raw target vectors. At
inference the predicted vector direction becomes the truthful direction and
its antipode the hallucination direction. The existing vMF gate (`kappa`,
`beta`) decides whether to intervene, and `alpha` controls the spherical
rotation of the hidden-state direction. The original hidden-state norm is
preserved.

This preserves the repository's previous spherical L2S inference behavior.
