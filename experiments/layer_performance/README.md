# Llama layer performance: S2S-L and S2S-R

This experiment scans Llama-3.1-8B-Instruct blocks **0–31** on TruthfulQA.
Each layer is trained independently with outer folds **0 and 1**. Both methods
share the direct-vector MLP checkpoint for a given layer/fold because their
training objectives are identical in this repository. The grid changes only
inference hyperparameters; training settings retain the repository defaults
(mean positives, hidden width 256, 2000 epochs, batch size 256, learning rate
0.001, seed 42, within-fold MLP fraction 0.5).

| Method | Parameter 1 | Parameter 2 |
|---|---|---|
| S2S-L (default `fast`) | strength: 1, 2 | slerp_t: 0.2, 0.6 |
| S2S-R (default `fast`) | weight_kappa: 3, 10 | margin: -0.7, -0.3 |

The launcher defaults to a small grid while keeping all 32 layers and both folds.
`SEARCH_PROFILE` controls the grid (counts assume both methods and all layers):

| Profile | strength | slerp_t | weight_kappa | margin | Fold evaluations |
|---|---|---|---|---|---|
| `fast` (default) | 1, 2 | 0.2, 0.6 | 3, 10 | -0.7, -0.3 | 512 |
| `balanced` | 1, 2, 3 | 0.2, 0.6, 1 | 1, 3, 10 | -0.7, -0.3, 0.3 | 1152 |
| `full` (original) | 0.5, 1, 1.5, 2, 2.5, 3 | 0, 0.2, 0.4, 0.6, 0.8, 1 | 1, 2, 3, 5, 10, 20 | -0.9, -0.7, -0.5, -0.3, 0, 0.3 | 4608 |

`fast` reduces evaluation count by 9×, `balanced` by 4×. Total wall time also
includes extraction/training/model loading, so these are not guaranteed runtime
speedups. Smaller grids may miss the original grid's optimum. For a layer
comparison, start with `fast`, then refine selected layers with `balanced` or
custom grids. These presets are original-grid subsets, preserving reuse of
matching cells from previous runs. Training settings stay unchanged to reuse MLPs.

For **each shared parameter tuple**, first compute the arithmetic mean of fold0
and fold1 MC1. Then select the tuple maximizing that mean at each layer, for each
method separately. MC2/MC3 are reported at the same MC1-selected tuple. Fold sizes
are not used as weights. Ties use the first numeric parameter tuple in sorted
order. The grids are selected directly on the held-out outer test questions:
all reports carry `selection=test_set_sweep_best`. These are exploratory test-set
sweep maxima, not validation-selected test estimates.

Run from the repository (no other preparation required):

```bash
GPU=0 bash experiments/layer_performance/run_llama.sh
```

The default launcher run prepares 32 feature files, trains **64** MLPs and evaluates
**512** fold-specific cells (32 × 2 methods × 4 tuples × 2 folds). One model load serves
both methods and both folds during each layer's grid evaluation. Feature
extraction is a separate model load per layer. It captures the actual block
output with a forward hook, so block 31 uses the state before final normalization,
at exactly the intervention location. It uses the existing standard TruthfulQA
feature prompts and existing zero-shot MC scoring/intervention implementations.
It does not modify `method/`.

Preview every command without creating files or loading a model:

```bash
DRY_RUN=1 bash experiments/layer_performance/run_llama.sh
```

Use a larger grid explicitly:

```bash
GPU=0 SEARCH_PROFILE=balanced bash experiments/layer_performance/run_llama.sh
GPU=0 SEARCH_PROFILE=full bash experiments/layer_performance/run_llama.sh
```

Stop any previous launcher and its running child before restarting in the same
output directory. Restarting reuses matching completed cells; an interrupted cell
is recomputed. Only shell/docs are changed for these presets, preserving the Python
source fingerprint used by existing caches. Summaries are rebuilt for the requested
grid; raw results outside that grid remain available. Calling `run.py` directly
still defaults to the original full grid; `SEARCH_PROFILE` applies to the launcher.

Restrict layers/methods or change the output root:

```bash
GPU=0 LAYERS="0 14 31" METHODS="S2S-L S2S-R" \
  OUTPUT_DIR=/path/to/results bash experiments/layer_performance/run_llama.sh
# Equivalent layer CLI: --layers 0,14,31 or --layers 0-31
```

`LLAMA_MODEL_DIR` overrides the model path; `PYTHON_BIN` selects Python.
`DEVICE`, `DTYPE`, `SEED`, `POSITIVE_MODE`, `VAL_FRACTION`, `MLP_HIDDEN`, `EPOCHS`,
`BATCH_SIZE`, and `LEARNING_RATE` override the shared training configuration.
Grid overrides are space-separated `STRENGTH_VALUES`, `SLERP_T_VALUES`,
`WEIGHT_KAPPA_VALUES`, and `MARGIN_VALUES`. CLI overrides are also available;
use `--margin="-0.9 -0.7 ..."` for lists beginning with a minus sign.
Both folds and test-split evaluation are required.

`RUN_EXTRACT=auto` and `RUN_TRAIN=auto` reuse matching caches, and rebuild missing
or mismatched ones; `0` requires matching caches, `1` forces rebuilding.
`REUSE_EXISTING=1` resumes completed evaluation cells; `0` recomputes them.
Features/checkpoints use metadata sidecars bound to configuration, source code,
and artifact size/modification time. Each evaluation records its checkpoint
identity and complete evaluation context. Changing a checkpoint invalidates
its previous evaluations. Only the currently requested grid/layers/methods enter
summaries. Every completed cell is saved atomically; summaries are updated after
each completed layer. Avoid concurrent writers to the same output directory.

Default output: `results/experiment/layer_performance/LLAMA/`:

- `run_config.json`: requested layers, grids and training settings.
- `features/layer*.npz`: raw answer activations, with cache sidecars.
- `checkpoints/layer*/fold*.npz`: shared MLPs, with cache sidecars.
- `raw/S2S-{L,R}/layer*/fold*/*.json`: individual metrics, settings, held-out question IDs and token counts.
- `summaries/all_grid_fold_mean.csv|json`: all parameter tuples, both fold scores and their means.
- `summaries/best_per_layer.csv|json`: one MC1-selected tuple per method/layer, with corresponding MC2/MC3.

CPU verification (uses a random tiny Llama and synthetic questions; downloads no model):

```bash
/root/miniconda3/envs/py310/bin/python -m unittest discover \
  -s experiments/layer_performance -p 'test_*.py' -v
```

S2S-L here uses the existing Dyn2D static SLERP mixture (`S2S_L_Static`), matching
the agreed `strength × slerp_t` grid and `dyn2d_ablation`; S2S-R uses the current
adaptive weights and margin gate (`S2S_R`).

Metrics are fractions in [0, 1]. The default launcher yields 256 fold-mean grid
rows and 64 best-per-layer rows; `full` yields 2304 grid rows. A partial layer run
contains only its requested layers.
