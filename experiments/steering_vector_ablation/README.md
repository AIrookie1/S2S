# steering_vector_ablation

This experiment evaluates exactly three layer-wise MC1 curves on the same question/fold setup:

1. **Linear + shared mean** — additive steering with the arithmetic mean of the raw pair vectors `z_pos_raw - z_neg_raw` from the training fold.
2. **Spherical + shared mean** — spherical steering with that same mean vector, converted to `mu_T = normalize(mean)` and `mu_H = -mu_T`.
3. **GT: neg -> pos** — no mean vector and no pair-vector steering. For each MC1 false answer, the target decoder layer's **final answer-token hidden state** is directly replaced by the exact paired `z_pos_raw`; the stored `z_neg_raw` is the pre-steering endpoint. The truthful answer is scored without intervention.

The old `linear_pair_specific` and `spherical_pair_specific` curves are intentionally not part of this ablation anymore.

## Important MC1 consequence of the requested GT endpoint

The repository's zero-shot MC1 scorer sums the log-probabilities of the answer tokens that are already present in the input. The GT edit above happens only at the hidden state of the **final answer token**. In a causal decoder, that edit can affect logits after that final token, but it cannot change the probabilities that were used to score the already-observed answer tokens. Therefore, under the current MC1 scoring definition, `GT: neg -> pos` is expected to equal the no-intervention MC1 baseline (up to numerical noise) and to be layer-independent. This is intentional here because the requested GT definition is the exact final-token endpoint replacement.

## Single feature file per layer

Each layer keeps one base feature file:

```text
features/<model>_layer<L>_dynSVD_pairs.npz
```

The GT condition requires `z_pos_raw` and `z_neg_raw` in that file. Current `method/dyn3d_sga/extract_features.py` writes both fields.

There are no `*_fold0.npz` or `*_fold1.npz` copies. `vector_utils.py` reconstructs fold membership directly from the base file's `q_indices` with the same deterministic `np.array_split` rule previously used to create fold-specific files. Thus the fold definition is unchanged; only the redundant on-disk copies are removed.

For the default two-fold setup, selecting fold 0 means split 0 is held out and the experiment's training questions come from split 1; selecting fold 1 uses split 0 as the training questions. Running both folds therefore covers the same questions as the previous materialized-fold implementation.

## Run

If the requested `*_dynSVD_pairs.npz` files already exist, the recommended path is direct evaluation:

```bash
EVAL_ONLY=1 bash experiments/steering_vector_ablation/run.sh
bash experiments/steering_vector_ablation/plot.sh
```

`EVAL_ONLY=1` skips the entire feature-preparation stage: it does not call `extract_features.py`, does not migrate malformed old filenames, and does not delete legacy fold files. It only checks that every requested `features/<model>_layer<L>_dynSVD_pairs.npz` exists and then invokes `evaluate.py`. For example, `LAYERS="1" EVAL_ONLY=1 ...` directly reuses `features/llama3.1-8B-Instruct_layer1_dynSVD_pairs.npz`.

After updating from the old four-curve experiment, **run `run.sh` again before `plot.sh`**. Existing JSON files containing only `linear_pair_specific`, `linear_shared_mean`, `spherical_pair_specific`, and `spherical_shared_mean` do not contain the new `gt_neg_to_pos` MC1 value and cannot be converted from aggregate metrics alone.

Without `EVAL_ONLY=1`, the default is still `RUN_EXTRACT=auto`: missing base pair files are extracted automatically. To force re-extraction:

```bash
RUN_EXTRACT=1 bash experiments/steering_vector_ablation/run.sh
```

To forbid extraction while still keeping the normal migration/cleanup preparation logic:

```bash
RUN_EXTRACT=0 bash experiments/steering_vector_ablation/run.sh
```

Use `EVAL_ONLY=1` instead when you want a true skip straight to evaluation. `EVAL_ONLY=1` and `RUN_EXTRACT=1` are intentionally rejected as contradictory.

Useful overrides include `MODEL_KEY=QWEN`, `LAYERS="14"`, `FOLDS="0 1"`, `NUM_FOLDS=2`, and `GPU=0`.

The runner removes legacy `*_fold0.npz` / `*_fold1.npz` duplicates for the selected layers by default. Set `CLEAN_LEGACY_FOLD_FEATURES=0` only if you explicitly want to keep old copies.

Outputs are written to `results/experiment/steering_vector_ablation/`. Each per-layer JSON now contains only these three MC1 metrics:

```text
linear_shared_mean
spherical_shared_mean
gt_neg_to_pos
```

Figures are named `steering_vector_ablation_mc1.{png,pdf,svg}` and contain exactly those three curves.
