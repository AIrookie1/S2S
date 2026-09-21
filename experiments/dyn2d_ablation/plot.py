"""Plot the Dyn2D MC1 grid as a confusion-matrix-style heatmap."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Dyn2D MC1 ablation heatmap")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--title", default="Dyn2D Ablation: TruthfulQA MC1 (fold mean)")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")

    strengths = sorted({float(row["strength"]) for row in rows})
    slerps = sorted({float(row["slerp_t"]) for row in rows})
    lookup = {
        (float(row["slerp_t"]), float(row["strength"])): float(row["MC1"])
        for row in rows
    }
    matrix = np.asarray(
        [[lookup.get((t, strength), np.nan) for strength in strengths] for t in slerps],
        dtype=float
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(
    np.arange(len(strengths)),
    labels=[_fmt(x) for x in strengths],
    fontsize=20
    )

    ax.set_yticks(
        np.arange(len(slerps)),
        labels=[_fmt(x) for x in slerps],
        fontsize=20
    )
    ax.set_xlabel("Steering strength", fontsize=20)
    ax.set_ylabel("SLERP coefficient", fontsize=20)
    # ax.set_title(args.title)

    finite = matrix[np.isfinite(matrix)]
    midpoint = float((finite.min() + finite.max()) / 2.0) if finite.size else 0.5
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                # Use black/white text only for legibility; the heatmap itself
                # keeps Matplotlib's default colormap.
                text_color = "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=20,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=20)
    colorbar.set_label("MC1")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if args.pdf:
        pdf = Path(args.pdf)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {output}")


if __name__ == "__main__":
    main()
