from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import figure_size, save_figure, setup_plot
from plot_specs import get_figure_spec, is_visible
from plot_theme import METHOD_STYLES, get_style

FIGURE_NAME = "steering_vector_ablation"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, default=Path("./results/experiment/steering_vector_ablation"))
    p.add_argument("--model_name", default="llama3.1-8B-Instruct")
    p.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("./results/experiment/steering_vector_ablation/steering_vector_ablation_mc1"),
    )
    a = p.parse_args()

    rows = []
    for path in sorted(a.input_dir.glob(f"{a.model_name}_layer*_mc1.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append((int(payload["layer"]), payload["MC1"]))
    if not rows:
        raise FileNotFoundError(f"No per-layer JSON files for {a.model_name} in {a.input_dir}")
    rows.sort()
    layers = [item[0] for item in rows]

    spec = get_figure_spec(FIGURE_NAME)
    labels = {
        "linear_shared_mean": spec["text"]["linear_shared_mean"],
        "gt_neg_to_pos": spec["text"]["gt_neg_to_pos"],
    }
    if is_visible(FIGURE_NAME, "spherical_shared_mean", False):
        labels["spherical_shared_mean"] = spec["text"]["spherical_shared_mean"]

    required = set(labels)
    missing = {
        layer: sorted(required.difference(metrics))
        for layer, metrics in rows
        if required.difference(metrics)
    }
    if missing:
        details = "; ".join(f"layer {layer}: {keys}" for layer, keys in sorted(missing.items()))
        raise KeyError(
            "Per-layer JSON files are missing requested steering-vector metrics "
            f"({details}). Re-run experiments/steering_vector_ablation/run.sh before plotting."
        )

    setup_plot(FIGURE_NAME)
    style = get_style(spec["style"])
    fig, ax = plt.subplots(figsize=figure_size(FIGURE_NAME))

    for key, label in labels.items():
        line_style = METHOD_STYLES[key]
        ax.plot(
            layers,
            [item[1][key] for item in rows],
            label=label,
            color=line_style["color"],
            marker=line_style["marker"],
            linewidth=style["line_width"],
            markersize=style["marker_size"],
            markeredgewidth=style["marker_edge_width"],
            markeredgecolor=line_style["color"],
            markerfacecolor=line_style["color"],
        )

    if is_visible(FIGURE_NAME, "xlabel", True):
        ax.set_xlabel(spec["text"]["xlabel"], labelpad=7)
    if is_visible(FIGURE_NAME, "ylabel", True):
        ax.set_ylabel(spec["text"]["ylabel"], labelpad=2)
    if is_visible(FIGURE_NAME, "title", False) and spec["text"].get("title"):
        ax.set_title(spec["text"]["title"])

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.margins(x=0.02, y=0.08)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style["axes_line_width"])
    ax.spines["bottom"].set_linewidth(style["axes_line_width"])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=style["tick_width"],
        length=style["tick_length"],
        pad=5,
    )

    if is_visible(FIGURE_NAME, "legend", True):
        legend_kwargs = {
            "frameon": False,
            **spec.get("legend", {}),
        }
        ax.legend(**legend_kwargs)

    fig.subplots_adjust(**spec["layout"])
    save_figure(fig, a.output_prefix, png_dpi=300)
    plt.close(fig)
    print(f"Saved {a.output_prefix}.[png|pdf|svg]")


if __name__ == "__main__":
    main()
