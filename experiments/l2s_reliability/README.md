# L2S vector reliability experiment

This experiment implements the paper-facing comparison

\[
\text{vector source}\in\{\text{Mean},\text{L2S}\}
\quad\times\quad
\text{steering geometry}\in\{\text{Linear},\text{Spherical}\}.
\]

It is intentionally separate from the repository's older `l2s_slerp` method.

## Four MC1 conditions

All four conditions use the same held-out TruthfulQA questions in each fold.

1. **Mean + Linear**
   - `v_mean = mean(v_T)` over exactly `l2s_train_q_indices`
   - `h' = h + 1.4 * v_mean`

2. **Mean + Spherical**
   - `mu_T = normalize(v_mean)`, `mu_H = -mu_T`
   - uses the exact `method/spherical_steering/steering.py` geometry

3. **L2S + Linear**
   - `v_hat = MLP(h)`
   - `h' = h + 1.4 * v_hat`
   - preserves the MLP's raw predicted magnitude

4. **L2S + Spherical**
   - `v_hat = MLP(h)`
   - `mu_T(h) = normalize(v_hat)`, `mu_H(h) = -mu_T(h)`
   - uses the same exact spherical geometry as condition 2
   - therefore retains the predicted direction while discarding raw MLP magnitude

A no-steering baseline is evaluated by default and appears only as a reference
line in panel (a), not as a fifth 2×2 condition.

## Reliability diagnostics

Both diagnostics compare the predicted vector against

\[
v_T=z^+_{\rm raw}-z^-_{\rm raw}.
\]

For **Mean**, `v_pred = v_mean`.
For **L2S**, `v_pred = MLP(z_neg_raw)`.

Direction fidelity:

\[
\cos(v_{\rm pred}, v_T).
\]

Magnitude fidelity:

\[
\frac{\left|\|v_{\rm pred}\|-\|v_T\|\right|}{\|v_T\|}.
\]

Pair-level values are averaged within each TruthfulQA question before plotting,
so multiple positive×negative answer pairs from one question are not treated as
independent samples.

## Main figure: two panels

`plot.py` now produces a compact **1×2** paper figure.

### (a) Held-out performance

The four Mean/L2S × Linear/Spherical MC1 conditions are shown as an interaction
plot, with fold-level points and an optional no-steering reference line.

### (b) Vector reliability

Cosine similarity and magnitude error are combined in one two-dimensional
question-level plot:

- x-axis: `Cosine similarity to v_T ↑`
- y-axis: `Relative norm error ↓`

Each point is one question. The four clouds are:

- Mean · Train
- Mean · Test
- L2S · Train
- L2S · Test

A larger open marker shows the centroid of each cloud. Better vector estimates
move toward the lower-right corner: higher directional agreement and lower
magnitude error.

The default canvas is `8.48 × 2.80 in`, preserving the requested `424:140`
aspect ratio. PNG is exported at 600 dpi; PDF and SVG are vector outputs.

## Full run

`CHECKPOINT_TEMPLATE` must contain the literal `{fold}` placeholder:

```bash
CHECKPOINT_TEMPLATE='checkpoints/l2s/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_l2s_fold{fold}.npz' \
MODEL='llama3.1-8B-Instruct' \
bash experiments/l2s_reliability/run.sh
```

If checkpoint-relative feature paths are invalid, override them with:

```bash
CHECKPOINT_TEMPLATE='checkpoints/l2s/truthfulqa/llama3.1-8B-Instruct/llama3.1-8B-Instruct_layer14_l2s_fold{fold}.npz' \
FEATURE_PATH='./features/llama_layer14_pairs.npz' \
MODEL='llama3.1-8B-Instruct' \
bash experiments/l2s_reliability/run.sh
```

## Plot existing results only

If evaluation has already produced

- `mc1.csv`
- `diagnostics_question.csv`

you can regenerate the figure **without loading the model or rerunning any
evaluation**:

```bash
bash experiments/l2s_reliability/run.sh --plot-only
```

For a non-default result directory:

```bash
bash experiments/l2s_reliability/run.sh \
  --plot-only \
  --output-dir ./results/experiment/l2s_reliability
```

You can also call the plotter directly:

```bash
python experiments/l2s_reliability/plot.py \
  --input_dir ./results/experiment/l2s_reliability \
  --output_prefix ./results/experiment/l2s_reliability/l2s_reliability
```

## Outputs

By default:

- `mc1.csv`
- `mc1_question.csv`
- `diagnostics_question.csv`
- `summary.json`
- `l2s_reliability.png`
- `l2s_reliability.pdf`
- `l2s_reliability.svg`
- `l2s_reliability_summary.csv`

Pass `--save_pair_diagnostics` to `evaluate.py` only if the full pair-level
audit table is needed.


## Separate figure outputs

The plotter now writes the two paper figures independently:

```text
l2s_reliability_performance.png
l2s_reliability_performance.pdf
l2s_reliability_performance.svg

l2s_reliability_vector_reliability.png
l2s_reliability_vector_reliability.pdf
l2s_reliability_vector_reliability.svg
```

No combined two-panel file is required.

To regenerate both figures from existing results without rerunning evaluation:

```bash
bash experiments/l2s_reliability/run.sh --plot-only
```

To regenerate only the performance figure:

```bash
bash experiments/l2s_reliability/run.sh \
  --plot-only \
  --figure performance
```

To regenerate only the vector-reliability figure:

```bash
bash experiments/l2s_reliability/run.sh \
  --plot-only \
  --figure reliability
```

The standalone sizes can also be changed directly:

```bash
python experiments/l2s_reliability/plot.py \
  --input_dir ./results/experiment/l2s_reliability \
  --output_prefix ./results/experiment/l2s_reliability/l2s_reliability \
  --performance_width 3.85 \
  --performance_height 2.80 \
  --reliability_width 4.65 \
  --reliability_height 2.80
```
