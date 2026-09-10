#!/usr/bin/env python3
"""Fig. 3a: Llama vs Qwen, individual fold0/fold1 MC1 curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

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
        description="Plot Fig. 3a with individual fold curves, without error bars."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE / "summaries",
        help="Directory containing {model}_fold{0,1}.json summaries.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "fig3a_svd_rank_mc1",
    )
    parser.add_argument("--num-folds", type=int, default=2)
    args = parser.parse_args()
    if args.num_folds < 1:
        parser.error("--num-folds must be at least 1")
    return args


def plot_fold_series(ax, series_list: list[dict]) -> None:
    """Plot raw folds using the original model-specific Nature styling.

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

    # Reuse the project's JSON loader. The original fold measurements are
    # already stored in series["fold_values"]; mean/SD are not plotted.
    series = load_fold_series(args.input_dir, "svd_rank", args.num_folds)

    fig, ax = plt.subplots(figsize=(4.9, 4.0))
    plot_fold_series(ax, series)

    x = series[0]["x"]
    ax.set_xlabel("SVD rank")
    ax.set_ylabel("MC1")
    ax.set_xticks(x)

    # Display the x-axis as 1–9 with a scientific-notation multiplier ×10^2.
    # The underlying x values remain unchanged (e.g. 100, 200, ..., 900).
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((2, 2))
    x_formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    finish_axis(ax)

    fig.tight_layout(pad=0.45)
    save_figure(fig, args.output_prefix)

    # Keep the existing CSV format for compatibility. It contains the raw
    # fold0/fold1 values plus mean/SD columns for reference only.
    csv_path = write_aggregate_csv(
        args.output_prefix, "svd_rank", series
    )
    plt.close(fig)

    print("Plotted individual fold curves (no error bars).")
    print("Fold0: filled markers; fold1: hollow markers.")
    print("X-axis shown as 1–9 with ×10^2 multiplier.")
    print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
    print(f"Saved plotted data: {csv_path}")


if __name__ == "__main__":
    main()
