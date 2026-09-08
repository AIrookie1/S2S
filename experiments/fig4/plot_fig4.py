#!/usr/bin/env python3
"""Plot Fig. 4: MC1 as a function of router validation fraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FIG4_DIR = Path(__file__).resolve().parent
METHOD_COLOR = "#0072B2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the Fig. 4 validation-fraction sweep.")
    parser.add_argument(
        "--input", type=Path, default=FIG4_DIR / "val_fraction_mc1.json"
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=FIG4_DIR / "fig4_val_fraction_mc1"
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 12.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def load_results(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = sorted(
        payload.get("results", []), key=lambda item: float(item["val_fraction"])
    )
    if not records:
        raise ValueError(f"No results found in {path}")
    fractions = [float(record["val_fraction"]) for record in records]
    mc1 = [float(record["MC1"]) for record in records]
    return fractions, mc1


def plot(fractions: list[float], mc1: list[float], output_prefix: Path) -> None:
    figure, ax = plt.subplots(figsize=(10.45, 3.15))
    ax.plot(
        fractions,
        mc1,
        color=METHOD_COLOR,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=METHOD_COLOR,
        markeredgewidth=1.5,
    )
    ax.set_xlabel("Validation fraction")
    ax.set_ylabel("MC1")
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{fraction:.1f}" for fraction in fractions])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.025, y=0.10)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_prefix.with_suffix(suffix), dpi=600)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    setup_style()
    fractions, mc1 = load_results(args.input)
    plot(fractions, mc1, args.output_prefix)
    print(f"Loaded: {args.input}")
    print(f"Validation fraction: {fractions}")
    print(f"MC1: {mc1}")
    print(f"Saved: {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
