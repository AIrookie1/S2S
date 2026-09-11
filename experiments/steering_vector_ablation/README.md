# steering_vector_ablation

Descriptive replacement for the former `fig1a` experiment. It evaluates four MC1 curves under the same question/fold setup:

1. Linear additive steering + exact raw answer-pair vector.
2. Linear additive steering + shared arithmetic mean vector.
3. Uploaded Spherical-Steering + exact raw answer-pair vector, using `mu_T=normalize(v)` and `mu_H=-mu_T`.
4. Uploaded Spherical-Steering + shared mean vector using the same antipodal conversion.

Every scored choice is intervened on, matching the uploaded Spherical MC scoring behavior. For a pair-specific raw vector, the truthful and false member of that pair are both scored with the same intervention; MC1 succeeds only if the truthful answer wins every pairwise comparison.

The uploaded `vector_utils.py` is used unchanged for pair loading/mapping. Dyn3D's base pair-feature NPZ does not contain fold metadata, so `prepare_fold_features.py` adds deterministic `train_q_indices`, `test_q_indices`, `fold_idx`, and `num_folds` fields without changing any original activation/vector arrays.

## Run

If `*_dynSVD_pairs.npz` files already exist for all requested layers:

```bash
bash experiments/steering_vector_ablation/run.sh
bash experiments/steering_vector_ablation/plot.sh
```

To extract the base pair-feature files first:

```bash
RUN_EXTRACT=1 bash experiments/steering_vector_ablation/run.sh
```

Useful overrides include `MODEL_KEY=QWEN`, `LAYERS="0 1 ... 31"`, `FOLDS="0 1"`, `GPU=0`, and `OVERWRITE_FOLDS=1`.

Outputs are written to `results/experiment/steering_vector_ablation/`. New figures are named `steering_vector_ablation_mc1.{png,pdf,svg}`.
