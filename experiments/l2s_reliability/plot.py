#!/usr/bin/env python3
"""Plot the L2S reliability experiment as two independent paper figures.

The default visual scale is matched to experiments/steering_vector_ablation
and the provided reference image:

* each standalone PNG: ~1024 × 720 at 300 dpi
* placing the two figures side-by-side: ~2048 × 720
* font/line/marker scale: Matplotlib ~10–11 pt publication scale
* primary palette: #0072B2 / #D55E00

Figure 1: held-out TruthfulQA MC1 for Mean/L2S × Linear/Spherical.
Figure 2: joint direction/magnitude vector reliability.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SOURCE_ORDER = ("Mean", "L2S")
GEOMETRY_ORDER = ("Linear", "Spherical")
SPLIT_ORDER = ("Train", "Test")

# Match the palette visible in the steering_vector_ablation reference figure.
BLUE = "#0072B2"
ORANGE = "#D55E00"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot the L2S reliability experiment as separate figures"
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./results/experiment/l2s_reliability"),
        help="Directory containing mc1.csv and diagnostics_question.csv",
    )
    p.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("./results/experiment/l2s_reliability/l2s_reliability"),
        help=(
            "Base output path. Files are saved as "
            "<prefix>_performance.* and <prefix>_vector_reliability.*"
        ),
    )
    p.add_argument(
        "--figure",
        choices=["all", "performance", "reliability"],
        default="all",
        help="Generate both figures or only one of them.",
    )

    # At 300 dpi these defaults produce 1024 × 720 PNGs.
    # Two images placed side-by-side are therefore ~2048 × 720, matching
    # the reference image scale very closely.
    p.add_argument("--performance_width", type=float, default=3.41334)
    p.add_argument("--performance_height", type=float, default=2.40)
    p.add_argument("--reliability_width", type=float, default=3.41334)
    p.add_argument("--reliability_height", type=float, default=2.40)
    p.add_argument("--png_dpi", type=int, default=300)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required result file not found: {path}\n"
            "Run evaluation first, or use --input_dir to point to existing results."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result file is empty: {path}")
    return rows


def setup_style() -> None:
    """Use the same visual scale as the steering_vector_ablation reference."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(top=False, right=False, pad=2.5)
    ax.grid(False)


def weighted_mc1(rows: list[dict[str, str]], condition: str) -> float:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ValueError(f"Missing MC rows for condition={condition!r}")
    weights = np.asarray([float(row["questions"]) for row in selected], dtype=float)
    values = np.asarray([float(row["MC1"]) for row in selected], dtype=float)
    return float(np.average(values, weights=weights))


def fold_mc1(rows: list[dict[str, str]], condition: str) -> list[tuple[int, float]]:
    return sorted(
        (int(row["fold"]), float(row["MC1"]))
        for row in rows
        if row["condition"] == condition
    )


def draw_performance(ax, rows: list[dict[str, str]]) -> None:
    """Draw the clean 2×2 MC1 interaction experiment."""
    x = np.asarray([0.0, 1.0])
    colors = {"Linear": BLUE, "Spherical": ORANGE}
    markers = {"Linear": "o", "Spherical": "s"}

    condition = {
        ("Mean", "Linear"): "mean_linear",
        ("Mean", "Spherical"): "mean_spherical",
        ("L2S", "Linear"): "l2s_linear",
        ("L2S", "Spherical"): "l2s_spherical",
    }

    all_values: list[float] = []
    pooled: dict[tuple[str, str], float] = {}

    for geometry in GEOMETRY_ORDER:
        means = []
        fold_series = []
        for source in SOURCE_ORDER:
            name = condition[(source, geometry)]
            mean = weighted_mc1(rows, name)
            pooled[(source, geometry)] = mean
            means.append(mean)

            fold_values = fold_mc1(rows, name)
            fold_series.append(fold_values)
            all_values.extend(value for _, value in fold_values)

        ax.plot(
            x,
            means,
            color=colors[geometry],
            marker=markers[geometry],
            markerfacecolor=colors[geometry],
            markeredgecolor=colors[geometry],
            markersize=6.2,
            linewidth=1.5,
            label=geometry,
            zorder=4,
        )

        for xi, fold_values in zip(x, fold_series):
            if len(fold_values) == 1:
                offsets = np.asarray([0.0])
            else:
                offsets = np.linspace(-0.035, 0.035, len(fold_values))
            ax.scatter(
                xi + offsets,
                [value for _, value in fold_values],
                s=18,
                alpha=0.40,
                marker=markers[geometry],
                color=colors[geometry],
                linewidths=0,
                zorder=3,
            )

    baseline_rows = [row for row in rows if row["condition"] == "baseline"]
    if baseline_rows:
        baseline = weighted_mc1(rows, "baseline")
        all_values.extend(float(row["MC1"]) for row in baseline_rows)
        ax.axhline(
            baseline,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            label="No steering",
            zorder=1,
        )

    delta = pooled[("L2S", "Spherical")] - pooled[("L2S", "Linear")]
    ax.text(
        0.98,
        0.035,
        rf"$\Delta_{{S-L}}$={delta:+.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
    )

    ax.set_xticks(x, ["Mean vector", "L2S vector"])
    ax.set_ylabel("MC1")
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        handletextpad=0.55,
        borderaxespad=0.10,
        labelspacing=0.30,
    )

    if all_values:
        lo = max(0.0, min(all_values) - 0.04)
        hi = min(1.0, max(all_values) + 0.04)
        if hi - lo < 0.12:
            center = 0.5 * (hi + lo)
            lo = max(0.0, center - 0.06)
            hi = min(1.0, center + 0.06)
        ax.set_ylim(lo, hi)

    # Slightly increase horizontal margins so the two x positions in
    # Figure 1 sit a bit closer to the center of the plotting area.
    ax.margins(x=0.24)
    clean_axis(ax)


def reliability_group(
    rows: list[dict[str, str]],
    source: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["vector_source"] == source and row["split"] == split
    ]
    if not selected:
        raise ValueError(f"Missing diagnostic rows for {source}/{split}")

    cosine = np.asarray(
        [float(row["cosine_similarity"]) for row in selected],
        dtype=float,
    )
    magnitude = np.asarray(
        [float(row["magnitude_relative_error"]) for row in selected],
        dtype=float,
    )
    if not np.isfinite(cosine).all() or not np.isfinite(magnitude).all():
        raise ValueError(f"Non-finite diagnostics for {source}/{split}")
    return cosine, magnitude


def draw_reliability(ax, rows: list[dict[str, str]]) -> None:
    """Draw direction and magnitude reliability in one 2-D diagnostic."""
    # Keep the original four distinct group colors from the earlier plot.
    # Mean·Train, Mean·Test, L2S·Train, L2S·Test
    # follow Matplotlib's default color cycle.
    group_color = {
        ("Mean", "Train"): "#1f77b4",
        ("Mean", "Test"): "#ff7f0e",
        ("L2S", "Train"): "#2ca02c",
        ("L2S", "Test"): "#d62728",
    }
    split_marker = {"Train": "o", "Test": "s"}

    all_x: list[float] = []
    all_y: list[float] = []

    for source in SOURCE_ORDER:
        for split in SPLIT_ORDER:
            x, y = reliability_group(rows, source, split)
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())

            color = group_color[(source, split)]
            marker = split_marker[split]

            ax.scatter(
                x,
                y,
                s=18,
                alpha=0.24,
                marker=marker,
                color=color,
                linewidths=0,
                label=f"{source} · {split}",
                zorder=2,
            )

            # Large open centroid marker makes train/test generalization shifts
            # visible even after the figure is scaled down for a paper.
            ax.scatter(
                [float(np.mean(x))],
                [float(np.mean(y))],
                s=70,
                marker=marker,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=5,
            )

    ax.axvline(
        0.0,
        color="0.55",
        linestyle=":",
        linewidth=1.0,
        alpha=0.70,
        zorder=0,
    )
    ax.axhline(
        1.0,
        color="0.55",
        linestyle=":",
        linewidth=1.0,
        alpha=0.70,
        zorder=0,
    )

    ax.set_xlabel(r"Cosine similarity to $v_T$")
    ax.set_ylabel("Relative norm error")

    ax.text(
        0.985,
        0.035,
        "better ↘",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
    )

    if all_x:
        x_min = max(-1.0, min(all_x) - 0.06)
        x_max = min(1.0, max(all_x) + 0.06)
        if x_max - x_min < 0.35:
            center = 0.5 * (x_min + x_max)
            x_min = max(-1.0, center - 0.175)
            x_max = min(1.0, center + 0.175)
        ax.set_xlim(x_min, x_max)

    if all_y:
        y_max = max(all_y)
        pad = max(0.08, 0.06 * y_max)
        ax.set_ylim(0.0, y_max + pad)

    # Direct four-group legend, preserving the original four colors.
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=group_color[("Mean", "Train")],
            markeredgecolor=group_color[("Mean", "Train")],
            markersize=5.5,
            label="Mean · Train",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=group_color[("Mean", "Test")],
            markeredgecolor=group_color[("Mean", "Test")],
            markersize=5.2,
            label="Mean · Test",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=group_color[("L2S", "Train")],
            markeredgecolor=group_color[("L2S", "Train")],
            markersize=5.5,
            label="L2S · Train",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor=group_color[("L2S", "Test")],
            markeredgecolor=group_color[("L2S", "Test")],
            markersize=5.2,
            label="L2S · Test",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        ncol=2,
        fontsize=8.5,
        columnspacing=0.65,
        handletextpad=0.32,
        borderaxespad=0.10,
        labelspacing=0.22,
    )
    clean_axis(ax)


def save_figure(fig, output_base: Path, *, png_dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = png_dpi
        fig.savefig(output_base.with_suffix(suffix), **kwargs)


def write_figure_summary(
    path: Path,
    mc_rows: list[dict[str, str]] | None,
    diagnostics: list[dict[str, str]] | None,
) -> None:
    rows: list[dict[str, Any]] = []

    if mc_rows is not None:
        for source in SOURCE_ORDER:
            for geometry in GEOMETRY_ORDER:
                condition = (
                    f"mean_{geometry.lower()}"
                    if source == "Mean"
                    else f"l2s_{geometry.lower()}"
                )
                rows.append(
                    {
                        "figure": "performance",
                        "group": source,
                        "subgroup": geometry,
                        "metric": "MC1",
                        "value": weighted_mc1(mc_rows, condition),
                        "n": sum(
                            int(row["questions"])
                            for row in mc_rows
                            if row["condition"] == condition
                        ),
                    }
                )

    if diagnostics is not None:
        for source in SOURCE_ORDER:
            for split in SPLIT_ORDER:
                x, y = reliability_group(diagnostics, source, split)
                rows.extend(
                    [
                        {
                            "figure": "vector_reliability",
                            "group": source,
                            "subgroup": split,
                            "metric": "cosine_similarity",
                            "value": float(np.mean(x)),
                            "n": len(x),
                        },
                        {
                            "figure": "vector_reliability",
                            "group": source,
                            "subgroup": split,
                            "metric": "magnitude_relative_error",
                            "value": float(np.mean(y)),
                            "n": len(y),
                        },
                    ]
                )

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["figure", "group", "subgroup", "metric", "value", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for name, value in (
        ("--performance_width", args.performance_width),
        ("--performance_height", args.performance_height),
        ("--reliability_width", args.reliability_width),
        ("--reliability_height", args.reliability_height),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.png_dpi <= 0:
        raise ValueError("--png_dpi must be positive")

    setup_style()

    need_performance = args.figure in ("all", "performance")
    need_reliability = args.figure in ("all", "reliability")

    mc_rows = read_csv(args.input_dir / "mc1.csv") if need_performance else None
    diagnostics = (
        read_csv(args.input_dir / "diagnostics_question.csv")
        if need_reliability
        else None
    )

    if need_performance:
        assert mc_rows is not None
        fig, ax = plt.subplots(
            figsize=(args.performance_width, args.performance_height)
        )
        draw_performance(ax, mc_rows)
        fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.94)
        save_figure(
            fig,
            args.output_prefix.with_name(
                args.output_prefix.name + "_performance"
            ),
            png_dpi=args.png_dpi,
        )
        plt.close(fig)

    if need_reliability:
        assert diagnostics is not None
        fig, ax = plt.subplots(
            figsize=(args.reliability_width, args.reliability_height)
        )
        draw_reliability(ax, diagnostics)
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.19, top=0.94)
        save_figure(
            fig,
            args.output_prefix.with_name(
                args.output_prefix.name + "_vector_reliability"
            ),
            png_dpi=args.png_dpi,
        )
        plt.close(fig)

    write_figure_summary(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        mc_rows,
        diagnostics,
    )

    if args.figure == "all":
        print(
            "Saved reference-scale separate figures:\n"
            f"  {args.output_prefix}_performance.[png|pdf|svg]\n"
            f"  {args.output_prefix}_vector_reliability.[png|pdf|svg]"
        )
    elif args.figure == "performance":
        print(f"Saved {args.output_prefix}_performance.[png|pdf|svg]")
    else:
        print(f"Saved {args.output_prefix}_vector_reliability.[png|pdf|svg]")


if __name__ == "__main__":
    main()
