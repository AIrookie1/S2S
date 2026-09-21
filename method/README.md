# Methods

- `dyn3d_sga/`: Similarity-Graph Adaptive 3D steering. A two-layer MLP receives `normalize(h)` and directly predicts the full `v_i = h_pos - h_neg`; inference uses `normalize(v_hat_i - v_avg)` as the third Dyn3D direction while preserving the original adaptive 3-way spherical intervention.
- `spherical_steering/`: replaced by the uploaded `Spherical-Steering-main` implementation. Its core spherical interpolation is in `steering.py`; TruthfulQA and generic evaluators use that implementation directly.
- `l2s/`: Learn-to-Steer direct MLP. It learns raw `v_T = z_pos_raw - z_neg_raw` from raw `z_neg` and applies the Spherical-Steering interpolation dynamically at inference time.
