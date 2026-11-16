# Table 3: S2S-R component ablation

This experiment implements the five-row TruthfulQA ablation agreed for the paper plus a mean-only adaptive-weight and margin-gate control, and **does not modify any file under `method/`**. The paper setting is Llama-3.1-8B-Instruct, layer 14, fold 0.

| Row | MLP | S2S | AW | Margin Gate | Intervention |
|---|---|---|---|---|---|
| 1 | x | x | x | x | `SLERP(normalize(h), normalize(v_avg), 0.4)` and restore `||h||` |
| 1b | x | x | y | y | `vavg_aw_gate`: mean-only adaptive weighting, single SLERP, and margin gating; no MLP prediction or residual |
| 2 | y | x | x | x | predict `v_hat` once from the segment's last position; `SLERP(normalize(h), normalize(v_hat), 0.4)` and restore `||h||` |
| 3 | y | y | x | x | current symmetric 2-SLERP over `X,V,R` with fixed `(alpha,beta,gamma)=(0.6,0.3,0.1)` |
| 4 | y | y | y | x | current `adaptive_3way_weights()` + current symmetric 2-SLERP, with every token steered (no gate) |
| 5 | y | y | y | y | delegates to the current `get_3d_dynamic_intervention()` unchanged, including the margin gate |

Here `X=normalize(h)`, `V=normalize(v_avg)`, and `R=normalize(v_hat-v_avg)`. Rows 2-5 preserve the current sequence-level predictor semantics: one prediction from the last position of the intervention segment is broadcast across that segment. The current adaptive-weight implementation is imported as-is; this experiment does not "fix" or reinterpret it.

For `vavg_aw_gate`, the existing graph is evaluated as `(alpha,b1,b2)=adaptive_3way_weights(X,V,V)`: both steering nodes carry the average direction, without calling the predictor. Their weights are merged into `beta=b1+b2`, with `gamma=0`. Each token's candidate is `SLERP(X,V,beta) * ||h||`, and is applied only when `beta > alpha + margin`. Thus `AW=y` and `Margin Gate=y`, while `MLP=x` and `S2S=x` (S2S denotes the symmetric three-direction composition). `single_slerp_t` is unused for this variant; `weight_kappa` and `margin` control it. Its single SLERP reuses the numerically stable Dyn2D primitive, including parallel/antipodal handling. Reported mean weights use the merged `(alpha,beta,0)` across all evaluated tokens, and the steering ratio counts only tokens passing the gate.

Because the merged mean weight is at least `2/3` for unit `X,V`, the inherited default `margin=-0.7` passes every token for this control. To study selective gating, set a stricter margin (for example `--margin 0.5`); report the chosen value with the results.

Run:

```bash
bash experiments/component_ablation/run.sh
```

The current `run.sh` is configured for fold 1 (the existing local setting); its default checkpoint is `checkpoints/dyn3d_sga/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_dynMLP_fold1_mean.npz`. Override it with `CHECKPOINT_PATH=...` if needed, and pass the matching `--expected_fold` when changing folds. `MAX_SAMPLES=N` is available only for a quick smoke run; leave it unset for the paper result. All six variants run by default.

Run only the new control into a separate directory, preserving existing result tables:

```bash
OUTPUT_DIR=results/experiment/component_ablation/llama3.1-8B-Instruct/fold1_vavg_aw_gate \
  bash experiments/component_ablation/run.sh --variants vavg_aw_gate
```

For a fold-0-only run matching the paper setting:

```bash
CHECKPOINT_PATH=checkpoints/dyn3d_sga/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_dynMLP_fold0_mean.npz \
OUTPUT_DIR=results/experiment/component_ablation/llama3.1-8B-Instruct/fold0_vavg_aw_gate \
  bash experiments/component_ablation/run.sh --expected_fold 0 --variants vavg_aw_gate
```

Outputs are written under `results/experiment/component_ablation/<model>/fold1/` by default, or the selected `OUTPUT_DIR`:

- `summary.json`
- `table3_component_ablation.csv`
- `table3_component_ablation.tex`
- `details/<variant>.json`

`Avg.` is exactly `(MC1 + MC2 + MC3) / 3`.

A run writes summary/CSV/LaTeX tables for its selected variants only; use the default full run to generate a six-row table. A single-variant run does not merge its metrics into an earlier table.

## Joint kappa / margin search for the mean-only control

```bash
bash experiments/component_ablation/run_vavg_sweep.sh
```

This separate runner defaults to fold 0, GPU 2 and the existing layer-14 checkpoint. It loads the language model once and evaluates only `vavg_aw_gate` over 64 combinations:

- `KAPPAS="0.1 0.3 0.5 1 2 3 5 10"`
- `MARGINS="-0.7 0.34 0.5 0.7 0.9 0.95 0.99 1"`

`-0.7` is the all-pass reference; `1` is the no-intervention reference. Positive thresholds above `1/3` cover selective gating. The selected configuration maximizes **MC1**, as requested. Ties use Avg, MC2, MC3 in descending order, then smaller kappa and margin. Selection is on the checkpoint's test split by default, matching the component table: the reported best is the best observed test score in this grid, rather than a separate held-out estimate after tuning. `EVAL_SPLIT=train` is also supported; it uses checkpoint training questions, not an independent validation split.

The output directory defaults to `results/experiment/component_ablation/llama3.1-8B-Instruct/fold0_vavg_grid/`:

- `grid_config.json`: grids and evaluation context (checkpoint/data/code hashes, prompts, fold and question IDs).
- `cells/kappa*_margin*.json`: one result per completed pair, including mean weights and steering ratio.
- `grid_summary.csv` / `grid_summary.json`: completed combinations ranked by MC1.
- `best_config.json`: current best kappa/margin and all metrics. Check `metadata.complete` before treating this as the final grid best.

Results and the best-so-far configuration are saved after every pair. Rerunning resumes completed matching cells. A changed grid/checkpoint/context requires a separate `OUTPUT_DIR`; incomplete/corrupt cells are recomputed. Detailed per-question results are optional via `SAVE_DETAILS=1`. Existing Table 3 files are untouched.

Customize the scan:

```bash
GPU=2 FOLD_IDX=1 KAPPAS="0.5 1 2 3" MARGINS="0.34 0.5 0.7 0.9" \
OUTPUT_DIR=results/experiment/component_ablation/llama3.1-8B-Instruct/fold1_vavg_custom \
  bash experiments/component_ablation/run_vavg_sweep.sh
```

For a smoke check, use `MAX_SAMPLES=1` and a separate `OUTPUT_DIR`; these scores are for debugging only. `SELECTION_METRIC` defaults to `MC1` and can also be `MC2`, `MC3`, or `Avg`. The grid runner supports the evaluator's command-line options, forwarded after its defaults.
