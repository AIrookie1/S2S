# Dyn2D strength × SLERP ablation

This experiment is fixed to **Llama, fold 0**, as requested.

Dyn2D is evaluated on the 6×6 grid:

- Steering strength: `0.5 1.0 1.5 2.0 2.5 3.0`
- SLERP coefficient: `0.0 0.2 0.4 0.6 0.8 1.0`

For each cell:

```text
V = normalize(v_avg)
R = normalize(v_hat - v_avg)
D = SLERP(V, R, slerp_t)
D_scaled = D * ||v_hat||_2
h_new = h + strength * D_scaled
```

Run:

```bash
bash experiments/dyn2d_ablation/run.sh
```

The runner reuses an existing feature file when available, trains the direct MLP
once for fold0, loads Llama/checkpoint once for the full 36-cell sweep, and
writes:

- `dyn2d_ablation_mc1.csv`
- `dyn2d_ablation_results.json`
- `dyn2d_ablation_mc1_heatmap.png`
- `dyn2d_ablation_mc1_heatmap.pdf`

Set `RUN_EXTRACT=1` or `RUN_TRAIN=1` to force those stages; `auto` is the
default for both.
