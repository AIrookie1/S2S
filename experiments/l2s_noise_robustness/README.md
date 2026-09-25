# L2S target-noise robustness

This experiment tests robustness to corrupted **MLP training targets only**.
For every method, the original target construction is preserved; the only
change is that the target actually used by the MLP loss is corrupted once
before optimization.

For each clean target `v_i`, training uses

`v_tilde_i = v_i + sigma * ||v_i||_2 * normalize(epsilon_i)`,
`epsilon_i ~ N(0, I)`.

Each target receives exactly one noise draw before the training loop. That
noisy target is then fixed for every epoch and mini-batch. The noise RNG is
controlled by `NOISE_SEED`; using the same seed across noise levels holds each
method's sampled noise directions fixed while only changing their magnitude.

The following remain clean and unchanged:

- MLP inputs and question-level splits.
- Each method's original clean target definition before corruption.
- `v_avg` / mean-split geometry.
- Held-out TruthfulQA examples.
- Inference steering rules.
- Generic inference residual-noise support elsewhere in the repository.

For this experiment specifically, inference residual noise is hard-pinned to
`0.0` in every evaluator invocation and is therefore never injected.

Default target-noise levels are `1.0, 2.0, 3.0, 4.0, 5.0`.
There is intentionally no `sigma=0` condition in the default sweep.

Methods: `L2S-Linear`, `L2S-SLERP`, `S2S_L_Static`, and `S2S_R`.
Metric: held-out TruthfulQA MC1.

Run:

```bash
FAMILY=LLAMA GPU=0 bash experiments/l2s_noise_robustness/run.sh
```

Useful overrides include `NOISE_LEVELS`, `NOISE_SEED`, `RUN_TRAIN`,
`FEATURE_PATH`, `DYN_FEATURE_PATH`, and `RESULT_DIR`.

`RUN_TRAIN=auto` only reuses a checkpoint when its saved target-noise sigma,
noise seed, and fixed-noise sampling metadata match the requested run. This
prevents silently reusing older clean or inference-noise checkpoints.

Outputs include one directory per noise level, `mc1_by_noise.csv`, and the
Nature-style `l2s_noise_robustness.png/.pdf` figure. Directory tags use a
canonical numeric form (`1.0 -> noise_1`, `1.5 -> noise_1p5`) so training and
collection always agree.
