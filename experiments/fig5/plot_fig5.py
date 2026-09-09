#!/usr/bin/env python3
"""Fig. 5: Llama vs Qwen, individual fold0/fold1 MC1 curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from plot_common1 import (  # noqa: E402
    MODEL_STYLES,
    finish_axis,
    load_fold_series,
    save_figure,
    setup_nature_style,
    write_aggregate_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 5 with individual fold curves, without error bars."
    )
    parser.add_argument("--input-dir", type=Path, default=HERE / "summaries")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "fig5_weight_kappa_mc1",
    )
    parser.add_argument("--num-folds", type=int, default=2)
    args = parser.parse_args()
    if args.num_folds < 1:
        parser.error("--num-folds must be at least 1")
    return args


def _format_kappa(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def plot_fold_series(ax, series_list: list[dict]) -> None:
    """Plot raw folds, retaining the original model-specific Nature styling.

    Model: color, marker shape and line style from MODEL_STYLES.
    Fold: filled marker for fold0, hollow marker for fold1.
    No averaging or error bars are used for the plotted y values.
    """
    for series in series_list:
        model = series["model"]
        style = MODEL_STYLES.get(
            model,
            {
                "color": "#4D4D4D",
                "marker": "o",
                "linestyle": "-",
            },
        )

        # load_fold_series already provides the original, unaggregated MC1:
        # fold_values.shape == (num_folds, num_kappa_points).
        for fold, y in enumerate(series["fold_values"]):
            ax.plot(
                series["x"],
                y,
                label=f"{series['label']} · fold{fold}",
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["color"] if fold == 0 else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                linewidth=1.5,
                markersize=4.5,
                zorder=3,
            )


def main() -> None:
    args = parse_args()
    setup_nature_style()

    # Reuse the project's JSON loader. Its fold_values contain the raw
    # fold0/fold1 measurements; mean and sd are NOT used for plotting.
    series = load_fold_series(args.input_dir, "weight_kappa", args.num_folds)

    fig, ax = plt.subplots(figsize=(4.9, 4.0))
    plot_fold_series(ax, series)

    x = series[0]["x"]
    ax.set_xscale("log")
    ax.set_xlabel(r"Steering strength $\kappa$")
    ax.set_ylabel("MC1")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_kappa(v) for v in x])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xticklabels(
    ["" if float(v) == 15 else _format_kappa(v) for v in x]
)
    finish_axis(ax)

    fig.tight_layout(pad=0.45)
    save_figure(fig, args.output_prefix)

    # Keep the existing CSV format for compatibility. It includes the
    # exact fold values as well as mean/SD columns; the latter are not
    # plotted and do not affect the figure.
    csv_path = write_aggregate_csv(
        args.output_prefix, "weight_kappa", series
    )
    plt.close(fig)

    print("Plotted individual fold curves (no error bars).")
    print("Fold0: filled markers; fold1: hollow markers.")
    print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
    print(f"Saved plotted data: {csv_path}")


if __name__ == "__main__":
    main()
