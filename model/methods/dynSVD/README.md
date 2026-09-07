# dynSVD (Dynamic SVD Twin-Steering)

For each answer pair, the extractor preserves the hidden-state activations
and forms v_k = z_k+ - z_k- directly, without normalizing either endpoint.
For the selected `FOLD_IDX`, only that fold's questions are used for training.
They are split by question into an 80% geometry-training subset and a 20%
router-validation subset. The other fold is excluded from all training
computations and is retained only as the held-out test set in the checkpoint.
Only the 80% subset computes v_avg and the residual SVD basis B; the 20% subset
fits the MLP router and ScaleNet. An MLP router maps
the pre-intervention state to a softmax distribution over the rank-r basis:

    d_l = normalize(softmax(router(h_l)) @ B)
    h_l' = h_l + alpha * v_avg + gamma * d_l

The router predicts the softmax coefficients and a separate ScaleNet predicts a positive scalar scale. The routed basis combination is L2-normalized before scaling, so ScaleNet controls its magnitude. Router fitting uses both validation z- and z+ inputs: z- keeps the original pair-difference target, while z+ has dynamic target 0, scale target 0, and zero direction target. Test questions are never used in fitting.
At evaluation time, a candidate hidden state is compared with the original
hidden state using cosine similarity. Steering is skipped when this similarity
is at or above the configurable threshold. Evaluation and the progress bar
report the current batch cosine and steering rate.

Run the complete pipeline:

    bash scripts/run_dynSVD.sh

Or run stages individually:

    bash scripts/run_dynSVD_feature_extract.sh
    FOLD_IDX=0 VAL_FRACTION=0.2 bash scripts/run_dynSVD_train.sh
    FOLD_IDX=0 bash scripts/run_dynSVD_eval.sh

Set FOLD_IDX=1 for the second fold. Set `VAL_FRACTION=0.2` to keep the default
8:2 within-fold split. Set `HIDDEN_COSINE_THRESHOLD` in the eval
script to skip near-unchanged interventions. Feature NPZ files store prompts, question
IDs, normalized/raw activations, pair indices, and split metadata. Checkpoints
store v_avg, B, router weights, geometry-training/router-validation/test
question IDs.

## SVD rank

The number of SVD basis vectors is controlled manually with `--rank`; the
current default is 4. The canonical training script explicitly uses `--rank 4`.
The old energy-based selection (`--energy 0.95`) is retained only as a deprecated
fallback when `rank=None` is passed through the Python API. After changing rank,
retrain each fold checkpoint before evaluation.

The training objective combines vector MSE, scale MSE, and cosine-direction loss:

```text
vector_mse  = MSE(dyn, target_dyn)
scale_mse   = MSE(scale, ||target_residual||_2)
cosine_loss = 1 - mean(cosine_similarity(dyn, target_dyn)) over z- samples
total_loss  = vector_mse + scale_loss_weight * scale_mse \
              + cosine_loss_weight * cosine_loss
```

The scale and cosine weights are configurable through
`--scale_loss_weight` and `--cosine_loss_weight`; the shell wrapper exposes
`SCALE_LOSS_WEIGHT` and `COSINE_LOSS_WEIGHT`.

## Positive-answer pairing modes

Pair construction is performed during training, so the feature extractor remains
unchanged. Select the mode with `POSITIVE_MODE` in the training script:

```bash
POSITIVE_MODE=mean FOLD_IDX=0 bash scripts/run_dynSVD_train.sh
POSITIVE_MODE=mc1  FOLD_IDX=0 bash scripts/run_dynSVD_train.sh
```

`mean` averages all MC2 truthful-answer activations within each question and
pairs that prototype with every MC2 hallucinated answer. `mc1` loads
`mc1_targets` during training, selects only its single truthful answer, and
pairs it with every MC2 hallucinated answer. Both modes use the same original
feature file produced by `run_dynSVD_feature_extract.sh`; their checkpoints are
saved with different mode suffixes.

## Auxiliary interpolation samples

The training wrapper exposes two controls for optional interpolation samples:

```bash
USE_DATA_MODE=2 AUX_INTERP_COUNT=3 FOLD_IDX=0 \
  bash scripts/run_dynSVD_train.sh
```

`USE_DATA_MODE` selects the router/ScaleNet inputs: `0` = z- only, `1` = z-
plus z+, `2` = z- plus auxiliary samples, and `3` = z- plus z+ plus auxiliary
samples. `AUX_INTERP_COUNT` is the number of independent random linear
interpolations generated for every z-/z+ pair (the default is 1). An auxiliary
input is `z- + beta * (z+ - z-)`, with a matching target that is scaled by
`1 - beta`.

Auxiliary samples are used only while optimizing the router and ScaleNet. They
never contribute to `v_avg` or the SVD residual basis, which are computed from
the original geometry-training pairs only. The selected count and resulting
auxiliary sample total are recorded in the checkpoint metrics.

## Eval-time residual strength

Evaluation does not use the trained ScaleNet value. The routed basis
combination is normalized to a direction and the intervention is formed as

```text
dynamic = v_avg + gamma * routed_direction
hidden' = hidden + alpha * dynamic
```

Set `GAMMA` in `scripts/run_dynSVD_eval.sh` (or in the environment) to control
the residual contribution, for example:

```bash
GAMMA=0.5 FOLD_IDX=0 bash scripts/run_dynSVD_eval.sh
```

The existing hidden-state cosine gate remains active and can still skip an
intervention when the candidate is too similar to the original hidden state.
