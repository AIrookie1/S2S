# Fig. 1a: layer-wise MC1 results

`evaluate_all_layers.py` repeats Fig. 1 for each transformer layer using the
same pair-specific and shared-mean interventions. It loads the model once and
writes `results/layer{N}_mc1.json` immediately after each layer. Each file
contains only the two MC1 values (plus minimal run metadata); no per-question
CSV files or figures are produced.

The evaluator expects one pair-feature NPZ per layer and fold, named by the
default template:

`features/llama3.1-8B-Instruct_layer{layer}_dynSVD_pairs_fold{fold}.npz`

Run (extract missing features automatically):

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/fig1a/run.sh
```

To use only existing feature files without extraction; this mode avoids the expensive extraction step.


```bash
EXTRACT_FEATURES=0 CUDA_VISIBLE_DEVICES=0 bash experiments/fig1a/run.sh
```

Override `LAYERS`, `FOLDS`, `MODEL_DIR`, `ALPHA`, `OUTPUT_DIR`, or
`FEATURE_TEMPLATE` as needed. If a layer is incomplete, the default is to
stop with an error; pass `--allow_missing` directly to the evaluator to skip
incomplete layers.
