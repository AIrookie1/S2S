# Generic multiple-choice evaluation

This directory keeps the historical generic dataset entry points.  The
maintained Dyn3D implementation now lives under `method/dyn3d_sga/generic/`
and uses a direct MLP predictor:

- input: `normalize(h_neg)`
- target: `v_i = h_pos - h_neg`
- mean direction: `v_avg = mean(v_i)` on the mean-training split
- dynamic residual: `normalize(v_hat_i - v_avg)`
- steering: the original Dyn3D adaptive 3-way spherical interpolation and
  margin gate are unchanged.

`generic/evaluate_generic.py` forwards to the maintained evaluator.
`generic/script/run_generic_dataset.sh` forwards to the maintained LLAMA or
QWEN launcher; set `MODEL_FAMILY=QWEN` when needed (default: `LLAMA`).
