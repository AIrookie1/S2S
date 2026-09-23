#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import figure_size, finish, load_fold_series, plot_mean_sd, save, setup_plot, write_csv

FIGURE_NAME = "weight_kappa_ablation"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=HERE.parents[1] / "results" / "experiment" / "weight_kappa_ablation" / "summaries")
    p.add_argument("--output-prefix", type=Path, default=HERE.parents[1] / "results" / "experiment" / "weight_kappa_ablation" / "weight_kappa_ablation_mc1")
    p.add_argument("--num-folds", type=int, default=2)
    a = p.parse_args()

    setup_plot(FIGURE_NAME)
    series = load_fold_series(a.input_dir, "weight_kappa", a.num_folds)
    fig, ax = plt.subplots(figsize=figure_size(FIGURE_NAME))
    plot_mean_sd(ax, series, figure_name=FIGURE_NAME)
    ax.set_xscale("log")
    finish(ax, figure_name=FIGURE_NAME)
    fig.tight_layout()
    save(fig, a.output_prefix)
    csv_path = write_csv(a.output_prefix, "weight_kappa", series)
    plt.close(fig)
    print(f"Saved {a.output_prefix} and {csv_path}")


if __name__ == "__main__":
    main()
