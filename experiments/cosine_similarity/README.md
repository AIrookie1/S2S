# Raw `v_T` cosine-similarity experiment

This experiment compares **L2S**, **L2S-SLERP**, **Standard Steer**, **Dyn3D**, **DynLearnedSLERP**, **dynMLP**, and
**SADI-hidden** against the same raw pairwise ground-truth steering vectors
produced by `method/dyn3d_sga/extract_features.py`.

The integrated SADI path currently supports **LLaMA only**.

## Ground truth and multiple correct answers

For every extracted positive/negative Cartesian pair:

```text
v_gt = v_T = z_pos_raw - z_neg_raw
```

TruthfulQA MC2 can contain multiple correct answers and multiple incorrect
answers for one question. `build_question_vectors` creates every
`positive × negative` pair.

The cosine metric now uses the **all-pairs arithmetic mean**:

```text
cos_i = cosine(v_pred_i, v_gt_i)
final cosine = mean_i(cos_i)
```

So if the same negative answer has several correct GTs, all of those GTs are
kept and each `(negative, positive)` pair contributes one cosine value. There
is **no nearest/best GT selection** and no max operation before the final mean.

`num_pairs` reports the number of Cartesian pair rows, while `num_negatives`
is kept as an informational count of unique negative answers.

## Outer folds

For each outer fold, checkpoint `outer_train_q_indices` is the experiment
train split and `test_q_indices` is the held-out split. L2S, Dyn3D, dynMLP,
and SADI split IDs must match exactly or evaluation fails. L2S-SLERP reuses the L2S checkpoint and therefore introduces no additional training split.

### L2S

```text
v_pred = L2S(z_neg_raw)
cosine(v_pred, v_gt)
```

### L2S-SLERP

This method adds **no new learned predictor**. It first runs the existing L2S:

```text
v_hat = L2S(z_neg_raw)
```

It then uses the same pair-level mean direction used by Dyn3D/dynMLP:

```text
v_avg = mean(v_T on geometry-train questions)
r_hat = normalize(v_hat - v_avg)
```

and applies adaptive SLERP to the direction, then restores the magnitude
predicted by L2S:

```text
hidden_dir = normalize(z_neg)
b = normalize(v_avg)
(w_hidden, w_mean, w_residual) = adaptive_3way_weights(hidden_dir, b, r_hat)
t_adaptive = w_residual / (w_mean + w_residual)
direction = slerp(b, r_hat, t_adaptive)
v_pred = ||v_hat||_2 * direction

cosine(v_pred, v_gt)
```

The adaptive weights are the same weights used by the current three-sphere
Dyn3D method. There is no manual `t` hyperparameter.

So the comparison between L2S and L2S-SLERP isolates the effect of
**residualization + adaptive spherical direction interpolation** while keeping
the exact same trained L2S MLP and its predicted magnitude:

```text
L2S       : z_neg -> MLP -> v_hat
L2S-SLERP : z_neg -> MLP -> v_hat -> adaptive mean/residual SLERP -> restore ||v_hat||
```

No new checkpoint is required for L2S-SLERP.

## Focused L2S / adaptive L2S-SLERP evaluation

Use the dedicated runner when only these two methods are needed:

```bash
bash experiments/cosine_similarity/run_l2s_slerp_adaptive.sh
```

There is no `alpha` in the cosine experiment and no `t` sweep. By default,
`WEIGHT_KAPPA=3`, matching the current three-sphere TruthfulQA adaptive-weight
configuration. The output JSON/CSV includes cosine results plus the observed
adaptive-`t` and three-weight statistics.

Override, for example:

```bash
WEIGHT_KAPPA=3 FOLDS="0 1" \
  bash experiments/cosine_similarity/run_l2s_slerp_adaptive.sh
```

### Standard Steer

The baseline mean is recomputed from all raw `v_T` pairs in the complete
outer-train fold:

```text
v_avg_outer = mean(v_T[outer_train])
cosine(v_avg_outer, v_gt)
```

### Dyn3D

For this experiment, Dyn3D is trained with `positive_mode=cartesian`, so its
training pairs are exactly the same raw positive-negative Cartesian pairs used
by L2S:

```text
z_neg = z_neg_raw
v_target = z_pos_raw - z_neg_raw = v_T
```

Multiple correct answers are **not averaged before Dyn3D training**. The
method-level `v_avg` is still retained: it is the mean steering direction
computed from the training pair targets and is part of Dyn3D's geometry.


Dyn3D training and checkpoints are unchanged. The cosine experiment still
reuses the checkpoint's `v_avg`, SVD basis, sparse router, and
`adaptive_3way_weights`, but the final steering-vector construction is changed
to a **single direct SLERP between the mean direction and the routed residual**.

For each raw negative hidden state:

```text
a = normalize(z_neg_raw)
residual = normalize(router(a) @ basis)
b = normalize(v_avg)
(alpha, beta, gamma) = adaptive_3way_weights(a, b, residual)

t = gamma / (beta + gamma)
v_pred = slerp(b, residual, t)

cosine(v_pred, v_gt)
```

Here `z_neg_raw` is used only to condition the router and the adaptive weights.
It is **not** one of the final SLERP endpoints.

The cosine experiment therefore does **not** use:

```text
two_slerp_sym(...)
candidate * ||z_neg_raw||
candidate - z_neg_raw
```

The direct SLERP output itself is the Dyn3D steering vector. The normal Dyn3D
margin gate is also disabled for this representation-similarity experiment.

### DynLearnedSLERP

This is the renamed version of the newly uploaded Dyn3D variant. It is trained
on the same raw Cartesian pairs as the other learned methods, but unlike
current Dyn3D it **learns the first SLERP amount** rather than deriving it from
`adaptive_3way_weights`.

```text
v_T = z_pos_raw - z_neg_raw
v_avg = mean(v_T on geometry-train questions)
r_target = normalize(v_T - v_avg)

a = normalize(z_neg_raw)
coeff = TopKSoftmax(Router(a))
r_hat = normalize(coeff @ basis)
t_hat = sigmoid(ResidualParamsNet(a))

v_pred = slerp(normalize(v_avg), r_hat, t_hat)
```

The Router intentionally keeps the uploaded **Top-K + softmax** formulation.
`ResidualParamsNet` receives the same normalized negative state but is a
separate network.

The uploaded training originally supervised `r_hat` and `t_hat` separately.
This integrated version keeps those losses and adds the requested joint loss
on the exact vector used in the cosine experiment:

```text
L_router = 1 - cosine(r_hat, r_target)
L_param  = MSE(t_hat, t_true)
L_joint  = 1 - cosine(v_pred, normalize(v_T))

L = 1.0 * L_router
  + 0.03 * L_param
  + 1.0 * L_joint
  + 0.001 * L_load_balance
```

The second vMF hidden-state steering stage from the uploaded generation code is
not included in this representation cosine score. Here the quantity being
tested is the learned first-SLERP steering direction itself.

### dynMLP

`dynMLP` is a controlled replacement of only Dyn3D's SVD/router residual
predictor. It uses the same Cartesian pair targets and the same fold split:

```text
v_T = z_pos_raw - z_neg_raw
v_avg = mean(v_T on geometry-train questions)
r_target = normalize(v_T - v_avg)
```

Its residual predictor is:

```text
a = normalize(z_neg_raw)
residual = normalize(MLP(a))
```

The MLP is `Linear -> GELU -> Linear` with hidden size 256 by default. Unlike
L2S, it does **not** predict `v_T` directly; it predicts Dyn3D's residual
direction. The final spherical construction is exactly the same direct SLERP
used by Dyn3D:

```text
b = normalize(v_avg)
(alpha, beta, gamma) = adaptive_3way_weights(a, b, residual)
t = gamma / (beta + gamma)
v_pred = slerp(b, residual, t)
cosine(v_pred, v_gt)
```

Therefore the main comparison is:

```text
Dyn3D : z_neg -> Router -> SVD basis -> residual -> SLERP
dynMLP: z_neg -> MLP                -> residual -> SLERP
```

This isolates the effect of the SVD subspace/router from the effect of the
SLERP construction itself.

### SADI hidden

The uploaded SADI hidden branch ranks MLP-output coordinates by the average
true-minus-false activation difference, then dynamically multiplies selected
current activations.

The integrated extractor records the target decoder layer's MLP output in
addition to its decoder-layer output. For selected unit `j`:

```text
v_pred[j] = (strength - 1) * mlp_z_neg[j]
v_pred[other units] = 0
cosine(v_pred, v_gt)
```

This is in the same decoder-layer vector space because the MLP output enters
that layer through the residual addition.

SADI selection is leakage-safe:

```text
geometry_train questions -> rank hidden units
validation questions     -> choose count and strength
held-out test questions  -> final evaluation only
```

Defaults reproduce the uploaded hidden-branch grids:

```text
count    = 1 2 4 6 8 10 12 14 16 18 20
strength = 5 10 15 ... 80
```

For a cosine-only objective, a shared positive `strength` changes magnitude
but not direction for a fixed mask, so strengths are theoretically tied.
The full validation grid is still evaluated as requested; exact ties choose
the smaller count/strength deterministically.

## Run

From the repository root:

```bash
bash experiments/cosine_similarity/run.sh
```

Defaults:

- `MODEL_KEY=LLAMA`
- `FOLDS="0 1"`
- `RUN_EXTRACT=auto`
- `RUN_TRAIN=auto`
- output: `results/experiment/cosine_similarity/`

`RUN_EXTRACT=auto` also checks whether an existing feature file contains the
new SADI MLP fields. An older feature file is automatically regenerated. If
`RUN_EXTRACT=0`, the script fails with an explicit message instead.

`RUN_TRAIN=auto` also checks SADI's cosine-GT metric metadata. A checkpoint
created with the nearest-correct-GT metric is automatically retrained so
validation and final evaluation both use the all-pairs mean rule.

Useful overrides:

```bash
RUN_EXTRACT=0 RUN_TRAIN=0 \
FEATURE_PATH=/path/to/features.npz \
L2S_CHECKPOINT_TEMPLATE='/path/l2s_fold{fold}.npz' \
DYN3D_CHECKPOINT_TEMPLATE='/path/dyn_fold{fold}_cartesian.npz' \
DYNLEARNEDSLERP_CHECKPOINT_TEMPLATE='/path/dynlearnedslerp_fold{fold}_cartesian.npz' \
DYNMLP_CHECKPOINT_TEMPLATE='/path/dynmlp_fold{fold}_cartesian.npz' \
SADI_CHECKPOINT_TEMPLATE='/path/sadi_fold{fold}.npz' \
bash experiments/cosine_similarity/run.sh

SADI_COUNTS="1 2 4 8 16" \
SADI_STRENGTHS="5 10 20" \
bash experiments/cosine_similarity/run.sh

DRY_RUN=1 bash experiments/cosine_similarity/run.sh
```

## Outputs

```text
results/experiment/cosine_similarity/
  cosine_similarity.json
  cosine_similarity.csv
```

CSV contains fold/split/method all-pairs mean cosine for L2S, L2S-SLERP, Standard Steer, Dyn3D, DynLearnedSLERP, dynMLP, and SADI, plus standard deviation,
min/max, Cartesian pair count, unique-negative count, and zero-vector counts. JSON also records SADI's selected
count/strength and validation cosine plus checkpoint/split metadata.

## Tests

```bash
pytest -q experiments/cosine_similarity/test_cosine_similarity.py
python tools/validate_dyn3d_setup.py
```

A lightweight synthetic end-to-end smoke test can be run without loading a
language model by creating a synthetic feature NPZ and training the four
learned checkpoints on CPU.
