# Figure 2: Dyn3D router heatmap

Figure 2 visualizes the actual router softmax values used by Dyn3D evaluation.
Rows are five evaluation questions sampled without replacement from the checkpoint test split. The sample is reproducible and controlled by `SEED`/`--seed`. For each question, the script finds the
global maximum router weight and plots a contiguous 32-vector window centered
around that peak (shifted at rank boundaries). This is not Top-K selection.
Each row label reports its actual `Vstart–Vend` range and peak vector. Color
intensity represents the router value.

The router input matches `method/dyn3d_sga/evaluate_3d_dyn.py`:

```python
weights = softmax(router.logits(raw_hidden_state))
```

Run:

```bash
bash experiments/fig2/run.sh
```

The default checkpoint is
`features/llama3.1-8B-Instruct_layer14_dynSVD_fold1_mean.npz`. Override it or
the selected questions when needed:

```bash
CHECKPOINTS="features/checkpoint.npz" NUM_QUESTIONS=5 NUM_VECTORS=32 SEED=42 \
  bash experiments/fig2/run.sh
```

Outputs in `experiments/fig2/figures/`:

- `fig2_router_weights.png`
- `fig2_router_weights.pdf`
- `fig2_router_weights.svg`
- `fig2_router_weights.csv`
- `fig2_router_weights.json`

The figure uses the same 10.45 × 3.15 inch canvas, sans-serif typography, and
600 dpi export settings as Fig.1a.
