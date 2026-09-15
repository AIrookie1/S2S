# Methods

- `dyn3d_sga/`: project Dyn3D / Similarity-Graph Adaptive 3D steering, including the Dyn6 generic-task pipeline under `generic/`.
- `spherical_steering/`: replaced by the uploaded `Spherical-Steering-main` implementation. Its core spherical interpolation is in `steering.py`; TruthfulQA and generic evaluators use that implementation directly.
- `l2s/`: Learn-to-Steer direct MLP. It learns raw `v_T = z_pos_raw - z_neg_raw` from raw `z_neg` and applies the Spherical-Steering interpolation dynamically at inference time.
