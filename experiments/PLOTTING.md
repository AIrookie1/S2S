# Experiment plotting configuration

All experiment figure styling is centralized in three files.

## 1. `plot_sizes.py` — physical figure size

The four standard sizes are built from two widths and two heights:

| preset | size (inch) |
|---|---:|
| `half_half` | `10 × 8` |
| `half_quarter` | `10 × 4` |
| `quarter_half` | `5 × 8` |
| `quarter_quarter` | `5 × 4` |

The current `10 × 4` and `5 × 4` ablation sizes are therefore preserved.
Special figures (`fig3d_adaptive_weights`, heatmap, L2S reliability, word-number, etc.)
are also defined in this file, so their custom aspect ratio remains editable
from one place.

## 2. `plot_theme.py` — global visual style

Edit this file to change shared values such as:

- base font size
- axis-label / tick-label / legend / annotation font sizes
- marker size and marker edge width
- line width
- error-bar line width, cap size and cap thickness
- tick width and length
- model/method colors
- density palette
- PNG DPI and vector-font settings

The four standard sizes each have their own style profile (`half_half`,
`half_quarter`, `quarter_half`, `quarter_quarter`). They currently start from
the two visual scales already used by the project, but can be tuned independently.
Special compact/word-number/heatmap figures have their own profiles in the same file.

## 3. `plot_specs.py` — per-figure text and layout

Edit this file for figure-specific behavior:

- which size/style preset a figure uses
- x/y labels and titles
- whether xlabel/ylabel/title/legend/annotation/value labels are shown
- legend location
- annotation coordinates
- subplot margins
- heatmap annotation format
- ternary-grid / mean-marker layout
- word-number figure-level text positions

For example, changing `fig3c_gate_margin["size"]` from `half_quarter` to
`half_half` changes its physical size without touching the plotting script.
Changing `fig3c_gate_margin["show"]["legend"]` to `False` removes its legend.

## Shared helpers

`plot_common.py` contains aggregation, common error-bar plotting, axis cleanup,
and saving. `plot_large.py` is kept only as a compatibility wrapper; new plots
should use `plot_common.py` plus the three configuration files above.

## `fig3d_adaptive_weights`

`experiments/fig3d_adaptive_weights/fig3d_plot.sh` uses the current S2S_R method and standard
checkpoint layout. Defaults are Llama fold0 and `kappa = 1 3 5 10`.

`WEIGHTS_GATE_MODE=all` visualizes every token for which adaptive weights were
computed. `WEIGHTS_GATE_MODE=steered` visualizes only tokens that pass the
current margin gate. The main evaluation metrics remain based on the original
all-token statistics; the gate switch only changes the histogram/mean written
to the kappa-view weight JSON.
