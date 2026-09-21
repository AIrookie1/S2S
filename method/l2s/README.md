# L2S (Learn-to-Steer)

L2S learns a direct vector predictor from raw answer-pair features:

```text
MLP(z_neg_raw) -> v_T = z_pos_raw - z_neg_raw
```

The training loss is ordinary MSE on the unnormalized input and target. The
L2 normalization is intentionally deferred to inference. For every hidden
state, the predicted vector is normalized to `mu_T`, its antipode is used as
`mu_H`, and the original Spherical-Steering vMF gate plus spherical
interpolation are applied while preserving the hidden state's norm.

The split metadata and launcher conventions match `method/dyn3d_sga`: the
outer fold is selected first, then the selected fold is split deterministically
by `val_fraction`; the L2S optimizer uses the same predictor-training subset as
Dyn3D's direct MLP.
