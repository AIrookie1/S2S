from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./results/experiment/steering_vector_ablation"),
    )
    p.add_argument(
        "--model_name",
        default="llama3.1-8B-Instruct",
    )
    p.add_argument(
        "--output_prefix",
        type=Path,
        default=Path(
            "./results/experiment/steering_vector_ablation/"
            "steering_vector_ablation_mc1"
        ),
    )
    a = p.parse_args()

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    rows = []

    for f in sorted(
        a.input_dir.glob(f"{a.model_name}_layer*_mc1.json")
    ):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append((int(d["layer"]), d["MC1"]))

    if not rows:
        raise FileNotFoundError(
            f"No per-layer JSON files for "
            f"{a.model_name} in {a.input_dir}"
        )

    rows.sort()
    layers = [x[0] for x in rows]

    labels = {
        "linear_shared_mean": "mean steer vector",
        # "spherical_shared_mean": "Spherical + shared mean",
        "gt_neg_to_pos": "pair-specific vector",
    }

    required = set(labels)
    missing = {
        layer: sorted(required.difference(metrics))
        for layer, metrics in rows
        if required.difference(metrics)
    }
    if missing:
        details = "; ".join(
            f"layer {layer}: {keys}" for layer, keys in sorted(missing.items())
        )
        raise KeyError(
            "Per-layer JSON files are missing the new three-curve metrics "
            f"({details}). Re-run experiments/steering_vector_ablation/run.sh "
            "before plotting."
        )

    # ------------------------------------------------------------
    # Plot style
    # Match the supplied reference figure
    # ------------------------------------------------------------
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],

        # Font sizes
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,

        # Axis appearance
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # Vector output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # Reference image is approximately 852 x 376 px
    fig, ax = plt.subplots(
        figsize=(8.52, 3.76),
        dpi=400,
    )

    # Okabe-Ito style colors
    styles = {
        "linear_shared_mean": {
            "color": "#0072B2",
            "marker": "o",
        },
        "spherical_shared_mean": {
            "color": "#D55E00",
            "marker": "s",
        },
        "gt_neg_to_pos": {
            "color": "#009E73",
            "marker": "^",
        },
    }

    for key, label in labels.items():
        ax.plot(
            layers,
            [x[1][key] for x in rows],
            label=label,
            color=styles[key]["color"],
            marker=styles[key]["marker"],

            # Reference-figure line/marker proportions
            linewidth=1.7,
            markersize=7.5,

            markeredgewidth=0.8,
            markeredgecolor=styles[key]["color"],
            markerfacecolor=styles[key]["color"],
        )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------
    ax.set_xlabel(
        "layer",
        fontsize=20,
        labelpad=7,
    )
    ax.set_ylabel(
        "MC1",
        fontsize=20,
        labelpad=2,
    )

    # Similar tick density to the reference figure:
    # 0, 4, 8, 12, ...
    ax.xaxis.set_major_locator(MultipleLocator(4))

    # Reference figure uses roughly 0.05 spacing
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    # Small horizontal margin, similar to reference
    ax.margins(x=0.02, y=0.08)

    # No grid in reference
    ax.grid(False)

    # Only left and bottom spines are visible
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Outward ticks
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=1.3,
        length=4,
        pad=5,
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=1,
        handlelength=2.0,
        handletextpad=0.7,
        labelspacing=0.5,
        borderaxespad=0.7,
    )

    # Explicit margins give a shape closer to the supplied image
    fig.subplots_adjust(
        left=0.115,
        right=0.988,
        bottom=0.225,
        top=0.97,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    a.output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        a.output_prefix.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        a.output_prefix.with_suffix(".pdf"),
    )

    fig.savefig(
        a.output_prefix.with_suffix(".svg"),
    )

    plt.close(fig)

    print(
        f"Saved {a.output_prefix}.[png|pdf|svg]"
    )


if __name__ == "__main__":
    main()