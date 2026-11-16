# Methods

- `S2S_R/`: S2S_R. A two-layer MLP receives `normalize(h)` and directly predicts the full `v_i = h_pos - h_neg`; inference uses `normalize(v_hat_i - v_avg)` as the third S2S_R direction while preserving the original adaptive 3-way spherical intervention.
- `spherical_steering/`: replaced by the uploaded `Spherical-Steering-main` implementation. Its core spherical interpolation is in `steering.py`; TruthfulQA and generic evaluators use that implementation directly.
- `l2s_linear/`: Learn-to-Steer with standard raw-scale additive steering `h_new = h + alpha * L2S(h)`.
- `l2s_slerp/`: Learn-to-Steer with the previous vMF-gated spherical/SLERP hidden-state rotation, preserving the hidden-state norm.
