"""Plot the Dyn2D MC1 grid as a confusion-matrix-style heatmap."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import figure_size, setup_plot
from plot_specs import get_figure_spec, is_visible
from plot_theme import SAVE_SETTINGS, get_style

FIGURE_NAME = "dyn2d_heatmap"


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Dyn2D MC1 ablation heatmap")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")

    strengths = sorted({float(row["strength"]) for row in rows})
    slerps = sorted({float(row["slerp_t"]) for row in rows})
    lookup = {(float(row["slerp_t"]), float(row["strength"])): float(row["MC1"]) for row in rows}
    matrix = np.asarray(
        [[lookup.get((t, strength), np.nan) for strength in strengths] for t in slerps],
        dtype=float,
    )

    spec = get_figure_spec(FIGURE_NAME)
    setup_plot(FIGURE_NAME)
    style = get_style(spec["style"])
    fig, ax = plt.subplots(figsize=figure_size(FIGURE_NAME))
    image = ax.imshow(matrix, aspect="auto", cmap=spec["heatmap"]["cmap"])
    ax.set_xticks(np.arange(len(strengths)), labels=[_fmt(x) for x in strengths])
    ax.set_yticks(np.arange(len(slerps)), labels=[_fmt(x) for x in slerps])

    if is_visible(FIGURE_NAME, "xlabel", True):
        ax.set_xlabel(spec["text"]["xlabel"])
    if is_visible(FIGURE_NAME, "ylabel", True):
        ax.set_ylabel(spec["text"]["ylabel"])
    title = args.title or spec["text"].get("title")
    if is_visible(FIGURE_NAME, "title", False) and title:
        ax.set_title(title)

    if is_visible(FIGURE_NAME, "cell_values", True):
        value_format = spec["heatmap"].get("value_format", ".3f")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isfinite(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=style["annotation_font_size"],
                    )

    if is_visible(FIGURE_NAME, "colorbar", True):
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.ax.tick_params(labelsize=style["tick_label_size"])
        colorbar.set_label(spec["text"]["colorbar"])

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches=SAVE_SETTINGS["bbox_inches"], facecolor="white")
    if args.pdf:
        pdf = Path(args.pdf)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches=SAVE_SETTINGS["bbox_inches"], facecolor="white")
    plt.close(fig)
    print(f"Saved heatmap: {output}")


if __name__ == "__main__":
    main()
