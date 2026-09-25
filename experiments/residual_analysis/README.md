# Residual-direction analysis: S2S-R and S2S-L-Static

This folder implements the same three residual-direction experiments for either
of the repository's two residual-bearing steering methods:

- `S2S_R` -> `method/dyn3d_sga`
- `S2S_L_Static` -> `method/dyn2d`

Select the method with the `METHOD` environment variable when using `run.sh`.
`S2S_R` remains the default.

Both methods reuse the same direct-vector MLP target and the same predicted
residual definition:

```text
h_last -> normalize -> direct MLP -> v_hat
r_pred = normalize(v_hat - v_avg)
```

The important difference is how that residual direction is used.

## Method-specific intervention geometry

### S2S-R

The selected hidden-state direction `a`, mean direction `b`, and residual
`c=r_pred` are combined using the released Dyn3D adaptive 3-way weights and
symmetric two-SLERP:

```text
(alpha, beta, gamma) = adaptive_3way_weights(a, b, c)
d_rot = two_slerp_sym(a, b, c, alpha, beta, gamma)
gate = max(beta, gamma) > alpha + margin
```

The edit is norm-preserving and the margin gate decides whether the rotated
hidden direction replaces the original one.

For `Shuffled residual` and `Oracle residual`, the active residual is replaced
*before* adaptive weighting, so each condition recomputes its own
`alpha/beta/gamma` and its own gate.

### S2S-L-Static

The released Dyn2D static method first combines the global mean and residual
with a fixed SLERP coefficient and then applies additive steering:

```text
D = SLERP(normalize(v_avg), active_residual, slerp_t)
h_new = h + strength * ||v_hat_recipient|| * D
```

`strength` and `slerp_t` come from
`hyperparams/dyn2d/<FAMILY>/truthfulqa.sh` by default.

There are no adaptive `alpha/beta/gamma` weights and no margin gate.  For
`Shuffled residual` and `Oracle residual`, only the residual **direction** is
replaced.  The current recipient candidate still supplies its own predicted
vector magnitude `||v_hat||`; donor magnitude is never transferred.  This keeps
the intervention focused on the residual-direction question.

## Mean-only baseline

The baseline now follows the steering family selected by `METHOD`.

### `METHOD=S2S_R`

`Mean only` remains the released `method/spherical_steering` baseline. Its
hyperparameters come from:

```text
hyperparams/spherical_steering/<FAMILY>/truthfulqa.sh
```

All spherical cross-validation prototype folds are loaded. Each evaluated
question is routed to the prototype fold whose `test_q_indices` contains that
question, so the baseline is cross-fitted.

### `METHOD=S2S_L_Static`

`Mean only` is standard **linear additive mean steering** at the Dyn2D layer:

```text
h_mean = h + strength * v_avg
```

`v_avg` is the complete mean correction vector stored in the direct-MLP
checkpoint, including its magnitude. `strength` is the same Dyn2D `STRENGTH`
loaded from:

```text
hyperparams/dyn2d/<FAMILY>/truthfulqa.sh
```

The linear baseline does **not** normalize `v_avg`, does not multiply it by a
sample-specific `||v_hat||`, and does not use any Spherical-Steering prototype,
`kappa`, `alpha`, `beta`, or vMF gate. This makes the baseline method-family
matched to S2S-L-Static's additive/linear intervention.

## Oracle used only for analysis

The script first runs an **unsteered pre-pass with the exact MC evaluation
prompt, instruction setting, layer, and answer candidate**.  For each incorrect
candidate `a-`, it chooses the single truthful hidden from the same question
with the largest cosine similarity:

```text
z_pos_star(q, a-) = argmax_z cos(h(q, a-), z),  z in truthful hiddens of q
v_oracle(q, a-)   = z_pos_star(q, a-) - h(q, a-)
r_oracle(q, a-)   = normalize(v_oracle(q, a-) - v_avg)
```

Panels (a,c) therefore use a **per-negative nearest truthful target**, not the
truthful centroid.  The geometry CSV records `positive_target_answer_index` and
`positive_target_cosine` for auditing.

Truthful candidates are needed only for the Oracle MC condition.  For those
rows, the truthful centroid target is retained so that a truthful candidate
does not trivially select itself and produce `v_oracle = 0`.

The captured `h(q,a)` is the last-token output of the intervention layer,
matching both S2S-R and S2S-L-Static's sequence-level MLP probe.  Oracle
quantities are analysis-only.

If either `v_hat - v_avg` or `v_oracle - v_avg` has near-zero norm, the
corresponding residual direction is undefined and that candidate is excluded
from Panels (a,c).  The counts are reported in `summary.json`.

For Oracle MC, an undefined oracle residual falls back to the global mean
residual direction:

- S2S-R then recomputes adaptive weights and its gate normally.
- S2S-L-Static applies its fixed mean/residual SLERP and additive rule normally.

## Panel (a): residual alignment

Population: held-out **incorrect-answer** candidates.

```text
Matched:
    cos(r_pred_i, r_oracle_i)

Category-shuffled:
    cos(r_pred_pi(i), r_oracle_i)
    donor is in the same TruthfulQA Category but a different question

Global-shuffled:
    same, but donor can be from any different question
```

Panel (a) is method-independent because both methods share the same direct MLP,
`v_avg`, and residual definition.  The default is 100 shuffle seeds.
Incorrect-answer rows are averaged within question before inference/statistics,
so questions with more incorrect answers do not receive more weight.

## Panel (b): residual swap intervention

Four full TruthfulQA MC conditions are evaluated:

1. `Mean only`
2. `Shuffled residual`
3. `Predicted residual`
4. `Oracle residual`

### With `METHOD=S2S_R`

```text
Predicted residual:
    c = r_pred
    adaptive weights + gate computed normally

Shuffled residual:
    c = r_pred from a different question
    adaptive weights + gate recomputed from the shuffled c

Oracle residual:
    c = r_oracle
    adaptive weights + gate recomputed from the oracle c
```

### With `METHOD=S2S_L_Static`

```text
Predicted residual:
    D = SLERP(mean, r_pred, slerp_t)
    h_new = h + strength * ||v_hat_recipient|| * D

Shuffled residual:
    D = SLERP(mean, r_pred_from_other_question, slerp_t)
    h_new = h + strength * ||v_hat_recipient|| * D

Oracle residual:
    D = SLERP(mean, r_oracle, slerp_t)
    h_new = h + strength * ||v_hat_recipient|| * D
```

The full-MC shuffle defaults to `MC_SHUFFLE_SEEDS=5`; use 100 for a strict
high-compute final run.  Shuffled seeds are averaged within question before the
question-level bootstrap.

The plotted metric is `Delta MC1 over mean-only (percentage points)` with a 95%
question-level bootstrap CI.

## Panel (c): residual quality -> geometric gain

Population: the same held-out incorrect-answer candidates as Panel (a).

For each incorrect candidate, the target is the **single same-question truthful
hidden with maximum cosine similarity to that candidate's unsteered hidden**.
Thus different negatives from the same question may select different truthful
targets. The selected method is applied exactly as above using its predicted
residual, and the Mean-only output follows the same family rule described
above.

For S2S-R:

```text
angular_gain =
    angle(cross_fitted_spherical_mean_output, z_pos_star_for_this_negative)
    - angle(S2S_R_predicted_output, z_pos_star_for_this_negative)
```

For S2S-L-Static:

```text
linear_mean_output = normalize(h + strength * v_avg)
angular_gain =
    angle(linear_mean_output, z_pos_star_for_this_negative)
    - angle(S2S_L_Static_predicted_output, z_pos_star_for_this_negative)
```

For S2S-R, `selected_method_predicted_output` is the real gated norm-preserving
S2S-R result.

For S2S-L-Static, it is the normalized direction of the real additive result:

```text
h + strength * ||v_hat|| * SLERP(mean, r_pred, slerp_t)
```

Positive gain means the selected residual-bearing method moves the edited
representation closer to its nearest truthful target than its method-family
Mean-only baseline (Spherical Steering for S2S-R; linear additive `v_avg` for
S2S-L-Static). Candidates are split into five equal-count bins by matched
residual-oracle alignment, averaged within question, then bootstrapped over
questions.

## Run S2S-R

Llama:

```bash
FAMILY=LLAMA METHOD=S2S_R \
bash experiments/residual_analysis/run.sh
```

Qwen:

```bash
FAMILY=QWEN METHOD=S2S_R \
bash experiments/residual_analysis/run.sh
```

The default checkpoint is loaded from:

```text
checkpoints/dyn3d_sga/truthfulqa/<MODEL>/...
```

and the S2S-R `WEIGHT_KAPPA` / `MARGIN` settings come from
`hyperparams/dyn3d_sga/<FAMILY>/truthfulqa.sh`.

## Run S2S-L-Static

This path no longer requires Spherical-Steering prototype files because its
Mean-only baseline is linear additive steering.


Llama:

```bash
FAMILY=LLAMA METHOD=S2S_L_Static \
bash experiments/residual_analysis/run.sh
```

Qwen:

```bash
FAMILY=QWEN METHOD=S2S_L_Static \
bash experiments/residual_analysis/run.sh
```

The default checkpoint is loaded from:

```text
checkpoints/dyn2d/truthfulqa/<MODEL>/...
```

and `STRENGTH` / `SLERP_T` come from
`hyperparams/dyn2d/<FAMILY>/truthfulqa.sh` (currently the repository defaults
are `strength=3.0`, `slerp_t=0.2`).

You can override them explicitly, for example:

```bash
FAMILY=LLAMA METHOD=S2S_L_Static STRENGTH=3.0 SLERP_T=0.2 \
bash experiments/residual_analysis/run.sh
```

## Cheap geometry-only smoke test

```bash
FAMILY=LLAMA METHOD=S2S_L_Static SKIP_MC=1 MAX_QUESTIONS=20 \
bash experiments/residual_analysis/run.sh
```

or replace `S2S_L_Static` with `S2S_R`.

## Plot existing outputs only

Outputs are method-separated by default to avoid accidental overwrite:

```text
results/experiment/residual_analysis/S2S_R/<MODEL>/fold<FOLD>/
results/experiment/residual_analysis/S2S_L_Static/<MODEL>/fold<FOLD>/
```

Example:

```bash
FAMILY=LLAMA METHOD=S2S_L_Static PLOT_ONLY=1 \
bash experiments/residual_analysis/run.sh
```

## Useful overrides

```bash
NUM_SHUFFLE_SEEDS=100
MC_SHUFFLE_SEEDS=5
BOOTSTRAP_SAMPLES=1000
QUINTILES=5
SAVE_MC_SCORES=1
GPU=0
SPHERICAL_PROTO_DIR=prototypes/spherical_steering/truthfulqa
```

`SPHERICAL_PROTO_DIR` and the Spherical-Steering parameters apply only when
`METHOD=S2S_R`; `run.sh` loads those defaults from
`hyperparams/spherical_steering/<FAMILY>/truthfulqa.sh`. With
`METHOD=S2S_L_Static`, the Mean-only baseline is linear and no spherical
prototype/configuration is required.

## Outputs

- `geometry_pairs.csv` — incorrect-answer matched alignment, method-specific
  geometry diagnostics, method-family Mean-only baseline diagnostics, angular
  errors/gain, and alignment quintile. `mean_baseline_kind` explicitly records
  `spherical_steering` or `linear_additive_v_avg`.
- `alignment_question.csv` — one row per question for Matched /
  Category-shuffled / Global-shuffled visualization.
- `alignment_shuffle_null.csv` — seed-level shuffle-null statistics.
- `alignment_summary.csv` — question-bootstrap mean and 95% CI.
- `quintile_summary.csv` — Panel (c) means and question-bootstrap 95% CI.
- `mc_question.csv` — per-question MC1/MC2/MC3 for all Panel (b)
  conditions/seeds and steering statistics.  S2S-R rows report
  `alpha/beta/gamma`; S2S-L-Static rows report `slerp_t`, recipient predicted
  vector norm, and applied steering norm.
- `mc_summary.csv` — Panel (b) aggregate metrics and paired Delta-MC1 bootstrap
  CI.
- `mc_scores.jsonl` — optional per-answer log-probabilities.
- `summary.json` — selected method, definitions, parameters, counts, shuffle
  diagnostics, and summaries.
- `residual_analysis.png/.pdf` — final three-panel figure.

All figure labels are in English and use the repository's centralized plotting
configuration.


## Figure output

`plot.py` now produces three independent Nature-style figures instead of a
single three-panel canvas:

- `residual_analysis_alignment.png/.pdf`
- `residual_analysis_swap.png/.pdf`
- `residual_analysis_geometric_gain.png/.pdf`

All three use the shared project plotting theme, centralized size preset, and
600-DPI/vector export settings.


## Cross-category cosine-nearest control

Panels 1 and 2 include an additional deterministic hard-shuffle control. For
each recipient residual `r_i`, the donor is restricted to a different semantic
category (and a different question) and selected by maximum cosine similarity:

`j* = argmax_{j: category_j != category_i, q_j != q_i} cos(r_i, r_j)`.

Panel 1 therefore reports four alignment conditions: Matched,
Category-shuffled, Global-shuffled, and Cross-category cosine-nearest. Panel 2
reports five intervention conditions: Mean only, Shuffled residual,
Cross-category cosine-nearest residual, Predicted residual, and Oracle residual.
The new control is deterministic; the existing random shuffle controls retain
their multi-seed averaging.
