# Dyn3D on General Multiple-Choice Benchmarks

This directory evaluates **Similarity-Graph Adaptive 3D Spherical Steering** on the same five benchmarks used by the original generic spherical-steering pipeline:

- `copa`
- `storycloze`
- `mmlu_global`
- `winogrande`
- `boolq`

## Evaluation compatibility

The conversion changes only the steering method. The original behavior is retained for:

- dataset sources and train/evaluation splits;
- dataset-specific prompt templates;
- option-only log-probability scoring;
- option-length normalization;
- argmax prediction and accuracy calculation;
- MMLU category balancing and sample ranges.

Training feature extraction and evaluation import their dataset-specific
templates from `prompts_generic.py`. Consequently, COPA, StoryCloze, MMLU,
Winogrande, and BoolQ use the same question text, option list, answer prefix,
option boundary, and MMLU chat template on both sides.

## Complete run

```bash
bash generic/quick_set_generic.sh
```

The predefined layers remain unchanged:

| Dataset | Layer |
|---|---:|
| BoolQ | 12 |
| COPA | 16 |
| StoryCloze | 27 |
| MMLU Global | 28 |
| Winogrande | 16 |

Edit the clearly grouped variables at the top of `quick_set_generic.sh` to change GPU, model path, Dyn3D training settings, adaptive weights, or gate behavior.

Each completed dataset writes its MC1 result directly under this directory as
`generic/<dataset>_mc_results.json` (for example,
`generic/copa_mc_results.json`).

## Manual run

### 1. Extract the original generic training features

```bash
python generic/get_activations_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze --split train --layer 27 \
  --save_dir ./features_generic
```

### 2. Train the Dyn3D checkpoint

```bash
python method/dyn3d_sga/train_dynSVD.py \
  --feature_path ./features_generic/llama3.1-8B-Instruct_storycloze_train_l27.npz \
  --output_path ./checkpoints_generic/llama3.1-8B-Instruct_storycloze_l27_dyn3d.npz \
  --positive_mode mean --num_folds 1 --fold_idx 0 \
  --energy 0.90 --router_hidden 256 --epochs 600 \
  --batch_size 256 --learning_rate 0.001 --val_fraction 0.20 --device cuda
```

`num_folds=1` deliberately uses the complete original training split for the method. Its deterministic internal validation subset is used for router fitting; the benchmark evaluation split remains untouched.

### 3. Run the unchanged benchmark evaluation with Dyn3D steering

```bash
python generic/evaluate_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze --layer 27 \
    --checkpoint_path ./checkpoints_generic/llama3.1-8B-Instruct_storycloze_l27_dyn3d.npz \
  --weight_kappa 5.0 --weight_temperature 0.1 --use_gate 1 \
  --output_path ./generic/storycloze_mc_results.json
```

Use `--disable_steering` for the original no-intervention baseline. Use `--use_gate 0` to apply Dyn3D steering to every active option token.

## Files

| File | Purpose |
|---|---|
| `get_activations_generic.py` | Extract the original dataset-specific training features |
| `evaluate_generic.py` | Preserve the original MC evaluation while applying Dyn3D |
| `utils_generic.py` | Original datasets, splits, prompts, and feature helpers |
| `prompts_generic.py` | Shared dataset-specific training/evaluation prompts |
| `quick_set_generic.sh` | Extract, train, and evaluate all five datasets |
