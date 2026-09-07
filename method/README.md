# Method modules

- `dyn3d_sga/` — **Similarity-Graph Adaptive 3D Steering**: the project method. It combines dynamic SVD residual routing with adaptive three-vector spherical interpolation.
- `spherical_steering/` — **Spherical Steering**: the lightweight baseline intervention used by TruthfulQA evaluators.

All method-specific Python entry points live under this directory. Shell launchers are grouped in `script/<method>/`.
