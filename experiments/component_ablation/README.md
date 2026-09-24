# Table 3: S2S-R component ablation

This experiment implements the five-row TruthfulQA ablation agreed for the paper and **does not modify any file under `method/`**. The official run is Llama-3.1-8B-Instruct, layer 14, fold 0 only.

| Row | MLP | S2S | AW | Margin Gate | Intervention |
|---|---|---|---|---|---|
| 1 | x | x | x | x | `SLERP(normalize(h), normalize(v_avg), 0.4)` and restore `||h||` |
| 2 | y | x | x | x | predict `v_hat` once from the segment's last position; `SLERP(normalize(h), normalize(v_hat), 0.4)` and restore `||h||` |
| 3 | y | y | x | x | current symmetric 2-SLERP over `X,V,R` with fixed `(alpha,beta,gamma)=(0.6,0.3,0.1)` |
| 4 | y | y | y | x | current `adaptive_3way_weights()` + current symmetric 2-SLERP, with every token steered (no gate) |
| 5 | y | y | y | y | delegates to the current `get_3d_dynamic_intervention()` unchanged, including the margin gate |

Here `X=normalize(h)`, `V=normalize(v_avg)`, and `R=normalize(v_hat-v_avg)`. Rows 2-5 preserve the current sequence-level predictor semantics: one prediction from the last position of the intervention segment is broadcast across that segment. The current adaptive-weight implementation is imported as-is; this experiment does not "fix" or reinterpret it.

Run:

```bash
bash experiments/component_ablation/run.sh
```

The default checkpoint is `checkpoints/dyn3d_sga/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_dynMLP_fold0_mean.npz`. Override it with `CHECKPOINT_PATH=...` if needed. `MAX_SAMPLES=N` is available only for a quick smoke run; leave it unset for the paper result.

Outputs are written under `results/experiment/component_ablation/<model>/fold0/`:

- `summary.json`
- `table3_component_ablation.csv`
- `table3_component_ablation.tex`
- `details/<variant>.json`

`Avg.` is exactly `(MC1 + MC2 + MC3) / 3`.
