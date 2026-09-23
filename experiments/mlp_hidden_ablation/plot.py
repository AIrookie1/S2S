#!/usr/bin/env python3
"""Plot Dyn3D MLP hidden-size ablation using fold0/fold1 mean ± SD."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import figure_size, finish, load_fold_series, plot_mean_sd, save, setup_plot, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE.parents[1] / "results" / "experiment" / "mlp_hidden_ablation" / "summaries",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE.parents[1] / "results" / "experiment" / "mlp_hidden_ablation" / "mlp_hidden_ablation_mc1",
    )
    parser.add_argument("--num-folds", type=int, default=2)
    args = parser.parse_args()

    setup_plot("mlp_hidden_ablation")
    series = load_fold_series(args.input_dir, "mlp_hidden", args.num_folds)
    figure, axis = plt.subplots(figsize=figure_size("mlp_hidden_ablation"))
    plot_mean_sd(axis, series, figure_name="mlp_hidden_ablation")
    axis.set_xscale("log", base=2)
    x_values = sorted({int(x) for item in series for x in item["x"]})
    axis.set_xticks(x_values, labels=[str(x) for x in x_values])
    xticks = SPEC.get("xticks")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
    finish(axis, figure_name="mlp_hidden_ablation")
    figure.tight_layout()
    save(figure, args.output_prefix)
    csv_path = write_csv(args.output_prefix, "mlp_hidden", series)
    plt.close(figure)
    print(f"Saved {args.output_prefix} and {csv_path}")


if __name__ == "__main__":
    main()
