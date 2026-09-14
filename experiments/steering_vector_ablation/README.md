# steering_vector_ablation

Descriptive replacement for the former `fig1a` experiment. It evaluates four MC1 curves under the same question/fold setup:

1. Linear additive steering + exact raw answer-pair vector.
2. Linear additive steering + shared arithmetic mean vector.
3. Uploaded Spherical-Steering + exact raw answer-pair vector, using `mu_T=normalize(v)` and `mu_H=-mu_T`.
4. Uploaded Spherical-Steering + shared mean vector using the same antipodal conversion.

Every scored choice is intervened on, matching the uploaded Spherical MC scoring behavior. For a pair-specific raw vector, the truthful and false member of that pair are both scored with the same intervention; MC1 succeeds only if the truthful answer wins every pairwise comparison.

## Single feature file per layer

Each layer keeps only one base feature file:

```text
features/<model>_layer<L>_dynSVD_pairs.npz
```

There are no `*_fold0.npz` or `*_fold1.npz` copies. `vector_utils.py` reconstructs fold membership directly from the base file's `q_indices` with the same deterministic `np.array_split` rule previously used to create fold-specific files. Thus the fold definition is unchanged; only the redundant on-disk copies are removed.

For the default two-fold setup, selecting fold 0 means split 0 is held out and the experiment's training questions come from split 1; selecting fold 1 uses split 0 as the training questions. Running both folds therefore covers the same questions as the previous materialized-fold implementation.

## Run

If the requested `*_dynSVD_pairs.npz` files already exist:

```bash
bash experiments/steering_vector_ablation/run.sh
bash experiments/steering_vector_ablation/plot.sh
```

By default `RUN_EXTRACT=auto`: missing base pair files are extracted automatically. To force re-extraction:

```bash
RUN_EXTRACT=1 bash experiments/steering_vector_ablation/run.sh
```

To forbid extraction and require all base pair files to exist:

```bash
RUN_EXTRACT=0 bash experiments/steering_vector_ablation/run.sh
```

Useful overrides include `MODEL_KEY=QWEN`, `LAYERS="14"`, `FOLDS="0 1"`, `NUM_FOLDS=2`, and `GPU=0`.

The runner removes legacy `*_fold0.npz` / `*_fold1.npz` duplicates for the selected layers by default. Set `CLEAN_LEGACY_FOLD_FEATURES=0` only if you explicitly want to keep old copies.

Outputs are written to `results/experiment/steering_vector_ablation/`. Figures are named `steering_vector_ablation_mc1.{png,pdf,svg}`.
