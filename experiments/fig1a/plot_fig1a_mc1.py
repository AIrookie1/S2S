#!/usr/bin/env python3
"""Plot Fig. 1a: pair_specific vs shared_mean MC1 across layers."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class LayerRecord:
    layer: int
    pair_specific: float
    shared_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 1a MC1 by layer.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing layer*_mc1.json files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "fig1a_mc1_nature",
        help="Output file prefix without extension.",
    )
    parser.add_argument(
        "--keep-last-layer",
        action="store_true",
        help="Keep the last layer instead of removing it.",
    )
    return parser.parse_args()


def setup_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_records(results_dir: Path) -> list[LayerRecord]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records: list[LayerRecord] = []
    for path in sorted(results_dir.glob("layer*_mc1.json")):
        match = re.search(r"layer(\d+)_mc1\.json$", path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric = payload.get("MC1") or payload.get("mc1") or payload.get("MC") or payload.get("mc")
        if not isinstance(metric, dict):
            raise ValueError(f"Missing MC1 payload in {path}")
        try:
            records.append(
                LayerRecord(
                    layer=int(payload.get("layer", match.group(1))),
                    pair_specific=float(metric["pair_specific"]),
                    shared_mean=float(metric["shared_mean"]),
                )
            )
        except KeyError as exc:
            raise KeyError(f"Missing field {exc} in {path}") from exc

    records.sort(key=lambda item: item.layer)
    if not records:
        raise FileNotFoundError(f"No layer*_mc1.json files found in {results_dir}")
    return records


def plot(records: list[LayerRecord], output_prefix: Path, keep_last_layer: bool) -> None:
    if len(records) < 2 and not keep_last_layer:
        raise ValueError("Need at least two layers to drop the last one.")

    if not keep_last_layer:
        records = records[:-1]

    layers = [item.layer for item in records]
    pair_specific = [item.pair_specific for item in records]
    shared_mean = [item.shared_mean for item in records]

    fig, ax = plt.subplots(figsize=(10.45, 3.15))
    ax.plot(
        layers,
        pair_specific,
        color="#0072B2",
        marker="o",
        label="pair-specific vector",
    )
    ax.plot(
        layers,
        shared_mean,
        color="#D55E00",
        marker="s",
        label="mean steer vector",
    )

    ax.set_xlabel("layer")
    ax.set_ylabel("MC1")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02, y=0.08)
    ax.legend(frameon=False, loc="best", handlelength=2.2)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output_prefix.with_suffix(ext), dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_nature_style()
    records = load_records(args.results_dir)
    plot(records, args.output_prefix, args.keep_last_layer)
    print(
        f"Saved {args.output_prefix.with_suffix('.pdf')} and "
        f"{args.output_prefix.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
