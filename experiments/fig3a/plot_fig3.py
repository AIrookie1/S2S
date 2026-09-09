#!/usr/bin/env python3
"""Fig. 3a: Llama vs Qwen, mean ± SD across fold0/fold1."""

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

from plot_common import (  # noqa: E402
    finish_axis,
    load_fold_series,
    plot_model_series,
    save_figure,
    setup_nature_style,
    write_aggregate_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 3a with fold error bars.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_nature_style()
    series = load_fold_series(args.input_dir, "svd_rank", args.num_folds)

    fig, ax = plt.subplots(figsize=(4.9, 3.15))
    plot_model_series(ax, series)

    x = series[0]["x"]
    ax.set_xlabel("SVD rank")
    ax.set_ylabel("MC1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    finish_axis(ax)

    fig.tight_layout(pad=0.45)
    save_figure(fig, args.output_prefix)
    csv_path = write_aggregate_csv(args.output_prefix, "svd_rank", series)
    plt.close(fig)

    print("Error bars: mean ± sample SD across fold0/fold1")
    print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
    print(f"Saved plotted data: {csv_path}")


if __name__ == "__main__":
    main()
