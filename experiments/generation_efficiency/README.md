# Table 4: generation efficiency on one RTX 3090

This benchmark compares the current implementations of:

- No Steering
- Spherical Steering
- S2S-L (`method/dyn2d`)
- S2S-R (`method/dyn3d_sga`)

It does **not** modify `method/`. Llama-3.1-8B-Instruct layer 14 and the S2S fold-0 held-out TruthfulQA questions are used by default.

## Timing protocol

Each final generation is greedy and forced to exactly 50 generated tokens, so early EOS cannot give one method a smaller denominator. Model/checkpoint/tokenizer loading is excluded. CUDA is synchronized around timed regions. The benchmark performs warm-up first and then 5 timed repeats; the table reports mean +- sample standard deviation in tokens/s.

Two views are emitted:

1. `generation_only`: No Steering and Spherical Steering time their single generation. S2S-L/R reuse an untimed Pass-1/2 cache; **cached hidden -> predictor controls is still timed as part of Pass 3**.
2. `end_to_end`: S2S-L/R time Pass 1 + Pass 2 + controls + Pass 3. The numerator remains only the final Pass-3 tokens. No Steering and Spherical Steering have no probe passes, so their rows are identical to `generation_only`.

Every repeat stores `pass1_seconds`, `pass2_seconds`, `pass3_seconds`, `total_seconds`, `generated_tokens`, and `tokens_per_sec`. Overhead is runtime overhead relative to No Steering:

`100 * (mean_method_runtime / mean_no_steering_runtime - 1)`.

## Fold alignment detail

The repository currently uses two different 2-fold conventions. S2S fold 0 trains on the first half and holds out the second half, while Spherical Steering fold 1 trains on the first half and holds out the second half. Therefore the default benchmark intentionally pairs **S2S fold 0 with Spherical fold 1**. `benchmark.py` verifies that their `test_q_indices` match and stops rather than silently benchmarking on leaked/mismatched data.

## Current hyperparameters used

- Spherical Steering: `kappa=20.0`, `alpha=0.7`, `beta=-0.15`
- S2S-L: `strength=3.0`, `slerp_t=0.2`
- S2S-R: `weight_kappa=3`, `margin=-0.7`
- prompt preset: `null`, with the current instruction enabled

Run on a single visible 3090:

```bash
GPU=0 bash experiments/generation_efficiency/run.sh
```

For a quick smoke benchmark only, use e.g. `MAX_SAMPLES=2 REPEATS=1`. Leave those overrides unset for the paper run.

Outputs:

- `summary.json` with raw per-repeat phase timings and metadata
- `table4_generation_only.csv` / `.tex`
- `table4_end_to_end.csv` / `.tex`
