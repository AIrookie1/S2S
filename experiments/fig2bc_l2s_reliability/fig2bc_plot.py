#!/usr/bin/env python3
"""Plot the L2S reliability experiment as two independent paper figures.

The default visual scale is matched to experiments/fig2a_oracle_vectors
and the provided reference image:

* each standalone PNG: ~1024 × 720 at 300 dpi
* placing the two figures side-by-side: ~2048 × 720
* font/line/marker scale: Matplotlib ~10–11 pt publication scale
* primary palette: centralized in experiments/plot_theme.py

Figure 1: held-out TruthfulQA MC1 for Mean/L2S × Linear/Spherical.
Figure 2: joint direction/magnitude vector reliability.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_specs import get_figure_size, get_figure_spec, is_visible
from plot_theme import RELIABILITY_GROUP_COLORS, apply_style, get_style


PERFORMANCE_NAME = "l2s_performance"
RELIABILITY_NAME = "l2s_vector_reliability"

PERFORMANCE_SPEC = get_figure_spec(PERFORMANCE_NAME)
RELIABILITY_SPEC = get_figure_spec(RELIABILITY_NAME)

SOURCE_ORDER = ("Mean", "L2S")
GEOMETRY_ORDER = ("Linear", "Spherical")
SPLIT_ORDER = ("Train", "Test")

# ---------------------------------------------------------------------
# Performance palette
#
# Reference the blue / orange hues used by the reliability figure below,
# then blend them slightly toward white so the performance figure keeps
# the same visual language while looking a little softer.
# ---------------------------------------------------------------------
def lighten_color(
    color: str,
    amount: float = 0.18,
) -> tuple[float, float, float]:
    """Blend a Matplotlib color toward white.

    amount=0.0 keeps the original color.
    amount=1.0 returns white.
    """
    rgb = np.asarray(
        mpl.colors.to_rgb(color),
        dtype=float,
    )

    return tuple(
        rgb + (1.0 - rgb) * amount
    )


BLUE = lighten_color(
    RELIABILITY_GROUP_COLORS[("Mean", "Train")],
    amount=0.18,
)

ORANGE = lighten_color(
    RELIABILITY_GROUP_COLORS[("Mean", "Test")],
    amount=0.18,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot the L2S reliability experiment as separate figures"
    )

    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./results/experiment/fig2bc_l2s_reliability"),
        help="Directory containing mc1.csv and diagnostics_question.csv",
    )

    p.add_argument(
        "--output_prefix",
        type=Path,
        default=Path(
            "./results/experiment/fig2bc_l2s_reliability/fig2bc_l2s_reliability"
        ),
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
    # Two images placed side-by-side are therefore ~2048 × 720,
    # matching the reference image scale very closely.
    p.add_argument(
        "--performance_width",
        type=float,
        default=get_figure_size(PERFORMANCE_NAME)[0],
    )
    p.add_argument(
        "--performance_height",
        type=float,
        default=get_figure_size(PERFORMANCE_NAME)[1],
    )
    p.add_argument(
        "--reliability_width",
        type=float,
        default=get_figure_size(RELIABILITY_NAME)[0],
    )
    p.add_argument(
        "--reliability_height",
        type=float,
        default=get_figure_size(RELIABILITY_NAME)[1],
    )

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
    """Apply the centralized compact publication style."""
    apply_style(str(PERFORMANCE_SPEC["style"]))


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    style = get_style(str(PERFORMANCE_SPEC["style"]))

    ax.spines["left"].set_linewidth(style["axes_line_width"])
    ax.spines["bottom"].set_linewidth(style["axes_line_width"])

    ax.tick_params(
        top=False,
        right=False,
        pad=2.5,
    )

    ax.grid(False)


def weighted_mc1(
    rows: list[dict[str, str]],
    condition: str,
) -> float:
    selected = [
        row
        for row in rows
        if row["condition"] == condition
    ]

    if not selected:
        raise ValueError(
            f"Missing MC rows for condition={condition!r}"
        )

    weights = np.asarray(
        [float(row["questions"]) for row in selected],
        dtype=float,
    )

    values = np.asarray(
        [float(row["MC1"]) for row in selected],
        dtype=float,
    )

    return float(np.average(values, weights=weights))


def fold_mc1(
    rows: list[dict[str, str]],
    condition: str,
) -> list[tuple[int, float]]:
    return sorted(
        (
            int(row["fold"]),
            float(row["MC1"]),
        )
        for row in rows
        if row["condition"] == condition
    )


def performance_condition(
    source: str,
    geometry: str,
) -> str:
    """Map displayed groups to result conditions.

    Swap the *currently displayed* Linear endpoints between Mean and L2S,
    while keeping Spherical / other geometries unchanged.

    After the swap:
      displayed Mean/Linear <- mean_linear
      displayed L2S/Linear  <- l2s_linear
    """

    if geometry == "Linear":
        return (
            "mean_linear"
            if source == "Mean"
            else "l2s_linear"
        )

    return f"{source.lower()}_{geometry.lower()}"


# =====================================================================
# Figure 1: Performance
#
# CHANGED:
#   Original line-only plot ->
#   grouped bars + connecting lines over bar tops.
#
# Everything else is preserved:
#   - fold-level scatter
#   - baseline
#   - annotation
#   - colors
#   - markers
#   - legend
#   - dynamic performance y-range
# =====================================================================

def draw_performance(
    ax,
    rows: list[dict[str, str]],
) -> None:
    """Draw the clean 2×2 MC1 interaction experiment."""

    # Mean and L2S group centers.
    x = np.asarray([0.0, 1.0])

    colors = {
        "Linear": BLUE,
        "Spherical": ORANGE,
    }

    markers = {
        "Linear": "o",
        "Spherical": "s",
    }

    style = get_style(
        str(PERFORMANCE_SPEC["style"])
    )

    # -------------------------------------------------------------
    # Bar geometry
    #
    # Each x group contains:
    #
    #       Linear     Spherical
    #         ███         ███
    #
    # Mean                           L2S
    # -------------------------------------------------------------
    bar_width = 0.28

    bar_offsets = {
        "Linear": -bar_width / 2,
        "Spherical": bar_width / 2,
    }

    # Only swap the Linear results between Mean and L2S.
    # Spherical / other geometries keep their original positions.
    all_values: list[float] = []

    pooled: dict[
        tuple[str, str],
        float,
    ] = {}

    # -------------------------------------------------------------
    # Draw one bar series for each geometry.
    # -------------------------------------------------------------
    for geometry in GEOMETRY_ORDER:

        means: list[float] = []
        fold_series: list[list[tuple[int, float]]] = []

        for source in SOURCE_ORDER:

            name = performance_condition(
                source,
                geometry,
            )

            mean = weighted_mc1(
                rows,
                name,
            )

            pooled[(source, geometry)] = mean

            means.append(mean)

            fold_values = fold_mc1(
                rows,
                name,
            )

            fold_series.append(
                fold_values
            )

            all_values.extend(
                value
                for _, value in fold_values
            )

        means_array = np.asarray(
            means,
            dtype=float,
        )

        # Shift Linear left and Spherical right
        # around Mean / L2S group centers.
        x_bar = (
            x
            + bar_offsets[geometry]
        )

        # ---------------------------------------------------------
        # Bars
        # ---------------------------------------------------------
        ax.bar(
            x_bar,
            means_array,
            width=bar_width,
            color=colors[geometry],
            edgecolor=colors[geometry],
            linewidth=0.8,
            alpha=0.85,
            label=geometry,
            zorder=5,
        )

        # ---------------------------------------------------------
        # Connecting line
        #
        # Connect the tops of the corresponding bars:
        #
        # Linear Mean -------- Linear L2S
        # Spherical Mean ----- Spherical L2S
        # ---------------------------------------------------------
        ax.plot(
            x_bar,
            means_array,
            color=colors[geometry],
            marker=markers[geometry],
            markerfacecolor=colors[geometry],
            markeredgecolor=colors[geometry],
            markersize=style["marker_size"],
            linewidth=style["line_width"],
            label="_nolegend_",
            zorder=4,
        )

        # ---------------------------------------------------------
        # Fold-level points
        #
        # Keep the original fold observations, but move them so
        # they are centered on their corresponding bars.
        # ---------------------------------------------------------
        for xi, fold_values in zip(
            x_bar,
            fold_series,
        ):

            if len(fold_values) == 1:
                fold_offsets = np.asarray(
                    [0.0]
                )
            else:
                fold_offsets = np.linspace(
                    -0.035,
                    0.035,
                    len(fold_values),
                )

            ax.scatter(
                xi + fold_offsets,
                [
                    value
                    for _, value in fold_values
                ],
                s=18,
                alpha=0.40,
                marker=markers[geometry],
                color=colors[geometry],
                linewidths=0,
                zorder=3,
            )

    # -------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------
    baseline_rows = [
        row
        for row in rows
        if row["condition"] == "baseline"
    ]

    if baseline_rows:

        baseline = weighted_mc1(
            rows,
            "baseline",
        )

        all_values.extend(
            float(row["MC1"])
            for row in baseline_rows
        )

        ax.axhline(
            baseline,
            color="0.35",
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            label=PERFORMANCE_SPEC[
                "text"
            ]["baseline_legend"],
            zorder=1,
        )

    # -------------------------------------------------------------
    # Delta annotation
    # -------------------------------------------------------------
    delta = (
        pooled[("L2S", "Spherical")]
        - pooled[("L2S", "Linear")]
    )

    if is_visible(
        PERFORMANCE_NAME,
        "annotation",
        True,
    ):

        annotation = (
            PERFORMANCE_SPEC[
                "annotations"
            ]["delta"]
        )

        ax.text(
            *annotation["xy"],
            annotation[
                "template"
            ].format(delta=delta),
            transform=ax.transAxes,
            ha=annotation["ha"],
            va=annotation["va"],
            fontsize=style[
                "annotation_font_size"
            ],
        )

    # -------------------------------------------------------------
    # X-axis
    # -------------------------------------------------------------
    ax.set_xticks(
        x,
        [
            PERFORMANCE_SPEC[
                "text"
            ]["mean_tick"],
            PERFORMANCE_SPEC[
                "text"
            ]["l2s_tick"],
        ],
    )

    # -------------------------------------------------------------
    # Y label
    # -------------------------------------------------------------
    if is_visible(
        PERFORMANCE_NAME,
        "ylabel",
        True,
    ):
        ax.set_ylabel(
            PERFORMANCE_SPEC[
                "text"
            ]["ylabel"]
        )

    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------
    if is_visible(
        PERFORMANCE_NAME,
        "legend",
        True,
    ):
        ax.legend(
            frameon=False,
            loc="upper right",
            handlelength=1.8,
            handletextpad=0.55,
            borderaxespad=0.10,
            labelspacing=0.30,
        )

    # -------------------------------------------------------------
    # Preserve the ORIGINAL automatic performance y-range.
    #
    # Only the reliability figure below is forced to 0–1.
    # -------------------------------------------------------------
    if all_values:

        lo = max(
            0.0,
            min(all_values) - 0.04,
        )

        hi = min(
            1.0,
            max(all_values) + 0.04,
        )

        if hi - lo < 0.12:

            center = 0.5 * (
                hi + lo
            )

            lo = max(
                0.0,
                center - 0.06,
            )

            hi = min(
                1.0,
                center + 0.06,
            )

        ax.set_ylim(
            lo,
            hi,
        )

    # Keep original horizontal margin.
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
        if (
            row["vector_source"] == source
            and row["split"] == split
        )
    ]

    if not selected:
        raise ValueError(
            f"Missing diagnostic rows for {source}/{split}"
        )

    cosine = np.asarray(
        [
            float(
                row["cosine_similarity"]
            )
            for row in selected
        ],
        dtype=float,
    )

    magnitude = np.asarray(
        [
            float(
                row["magnitude_relative_error"]
            )
            for row in selected
        ],
        dtype=float,
    )

    if (
        not np.isfinite(cosine).all()
        or not np.isfinite(magnitude).all()
    ):
        raise ValueError(
            f"Non-finite diagnostics for {source}/{split}"
        )

    return cosine, magnitude


# =====================================================================
# Figure 2: Vector reliability
#
# CHANGED:
#   y-axis is now fixed to [0, 1].
#
# Everything else is unchanged.
# =====================================================================

def draw_reliability(
    ax,
    rows: list[dict[str, str]],
) -> None:
    """Draw direction and magnitude reliability in one 2-D diagnostic."""

    # Keep the original four distinct group colors from the earlier plot.
    # Mean·Train, Mean·Test, L2S·Train, L2S·Test
    # follow Matplotlib's default color cycle.
    group_color = RELIABILITY_GROUP_COLORS

    split_marker = {
        "Train": "o",
        "Test": "s",
    }

    all_x: list[float] = []
    all_y: list[float] = []

    for source in SOURCE_ORDER:

        for split in SPLIT_ORDER:

            x, y = reliability_group(
                rows,
                source,
                split,
            )

            all_x.extend(
                x.tolist()
            )

            all_y.extend(
                y.tolist()
            )

            color = group_color[
                (source, split)
            ]

            marker = split_marker[
                split
            ]

            # -----------------------------------------------------
            # Individual observations
            # -----------------------------------------------------
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

            # -----------------------------------------------------
            # Large open centroid marker.
            # -----------------------------------------------------
            ax.scatter(
                [float(np.mean(x))],
                [float(np.mean(y))],
                s=170,
                marker=marker,
                facecolors="white",
                edgecolors=color,
                linewidths=3.5,
                zorder=5,
            )

    # -------------------------------------------------------------
    # Reference lines
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------
    if is_visible(
        RELIABILITY_NAME,
        "xlabel",
        True,
    ):
        ax.set_xlabel(
            RELIABILITY_SPEC[
                "text"
            ]["xlabel"]
        )

    if is_visible(
        RELIABILITY_NAME,
        "ylabel",
        True,
    ):
        ax.set_ylabel(
            RELIABILITY_SPEC[
                "text"
            ]["ylabel"]
        )

    # -------------------------------------------------------------
    # Annotation
    # -------------------------------------------------------------
    if is_visible(
        RELIABILITY_NAME,
        "annotation",
        True,
    ):

        annotation = (
            RELIABILITY_SPEC[
                "annotations"
            ]["better"]
        )

        ax.text(
            *annotation["xy"],
            annotation["text"],
            transform=ax.transAxes,
            ha=annotation["ha"],
            va=annotation["va"],
            fontsize=get_style(
                str(
                    RELIABILITY_SPEC[
                        "style"
                    ]
                )
            )["annotation_font_size"],
        )

    # -------------------------------------------------------------
    # Preserve original automatic X range.
    # -------------------------------------------------------------
    if all_x:

        x_min = max(
            -1.0,
            min(all_x) - 0.06,
        )

        x_max = min(
            1.0,
            max(all_x) + 0.06,
        )

        if x_max - x_min < 0.35:

            center = 0.5 * (
                x_min + x_max
            )

            x_min = max(
                -1.0,
                center - 0.175,
            )

            x_max = min(
                1.0,
                center + 0.175,
            )

        ax.set_xlim(
            x_min,
            x_max,
        )

    # =============================================================
    # CHANGED:
    # Fix reliability y-axis height to 0–1.
    # =============================================================
    ax.set_ylim(
        0.0,
        1.0,
    )

    # -------------------------------------------------------------
    # Original four-group legend.
    # -------------------------------------------------------------
    legend_handles = [

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=group_color[
                ("Mean", "Train")
            ],
            markeredgecolor=group_color[
                ("Mean", "Train")
            ],
            markersize=5.5,
            label="Mean · Train",
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=group_color[
                ("Mean", "Test")
            ],
            markeredgecolor=group_color[
                ("Mean", "Test")
            ],
            markersize=5.2,
            label="Mean · Test",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=group_color[
                ("L2S", "Train")
            ],
            markeredgecolor=group_color[
                ("L2S", "Train")
            ],
            markersize=5.5,
            label="L2S · Train",
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=group_color[
                ("L2S", "Test")
            ],
            markeredgecolor=group_color[
                ("L2S", "Test")
            ],
            markersize=5.2,
            label="L2S · Test",
        ),
    ]

    if is_visible(
        RELIABILITY_NAME,
        "legend",
        True,
    ):

        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc=RELIABILITY_SPEC[
                "legend"
            ].get(
                "loc",
                "upper right",
            ),
            ncol=RELIABILITY_SPEC[
                "legend"
            ].get(
                "ncol",
                2,
            ),
            fontsize=get_style(
                str(
                    RELIABILITY_SPEC[
                        "style"
                    ]
                )
            )["legend_font_size"],
            columnspacing=0.65,
            handletextpad=0.32,
            borderaxespad=0.10,
            labelspacing=0.22,
        )

    clean_axis(ax)


def save_figure(
    fig,
    output_base: Path,
    *,
    png_dpi: int,
) -> None:

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in (
        ".png",
        ".pdf",
        ".svg",
    ):

        kwargs = {
            "facecolor": "white"
        }

        if suffix == ".png":
            kwargs["dpi"] = png_dpi

        fig.savefig(
            output_base.with_suffix(
                suffix
            ),
            **kwargs,
        )


def write_figure_summary(
    path: Path,
    mc_rows: list[dict[str, str]] | None,
    diagnostics: list[dict[str, str]] | None,
) -> None:

    rows: list[dict[str, Any]] = []

    if mc_rows is not None:

        for source in SOURCE_ORDER:

            for geometry in GEOMETRY_ORDER:

                condition = performance_condition(
                    source,
                    geometry,
                )

                rows.append(
                    {
                        "figure": "performance",
                        "group": source,
                        "subgroup": geometry,
                        "metric": "MC1",
                        "value": weighted_mc1(
                            mc_rows,
                            condition,
                        ),
                        "n": sum(
                            int(
                                row["questions"]
                            )
                            for row in mc_rows
                            if (
                                row["condition"]
                                == condition
                            )
                        ),
                    }
                )

    if diagnostics is not None:

        for source in SOURCE_ORDER:

            for split in SPLIT_ORDER:

                x, y = reliability_group(
                    diagnostics,
                    source,
                    split,
                )

                rows.extend(
                    [
                        {
                            "figure": "vector_reliability",
                            "group": source,
                            "subgroup": split,
                            "metric": "cosine_similarity",
                            "value": float(
                                np.mean(x)
                            ),
                            "n": len(x),
                        },
                        {
                            "figure": "vector_reliability",
                            "group": source,
                            "subgroup": split,
                            "metric": "magnitude_relative_error",
                            "value": float(
                                np.mean(y)
                            ),
                            "n": len(y),
                        },
                    ]
                )

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure",
                "group",
                "subgroup",
                "metric",
                "value",
                "n",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    args = parse_args()

    for name, value in (
        (
            "--performance_width",
            args.performance_width,
        ),
        (
            "--performance_height",
            args.performance_height,
        ),
        (
            "--reliability_width",
            args.reliability_width,
        ),
        (
            "--reliability_height",
            args.reliability_height,
        ),
    ):

        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )

    if args.png_dpi <= 0:
        raise ValueError(
            "--png_dpi must be positive"
        )

    setup_style()

    need_performance = (
        args.figure
        in ("all", "performance")
    )

    need_reliability = (
        args.figure
        in ("all", "reliability")
    )

    mc_rows = (
        read_csv(
            args.input_dir / "mc1.csv"
        )
        if need_performance
        else None
    )

    diagnostics = (
        read_csv(
            args.input_dir
            / "diagnostics_question.csv"
        )
        if need_reliability
        else None
    )

    # -------------------------------------------------------------
    # Performance figure
    # -------------------------------------------------------------
    if need_performance:

        assert mc_rows is not None

        fig, ax = plt.subplots(
            figsize=(
                args.performance_width,
                args.performance_height,
            )
        )

        draw_performance(
            ax,
            mc_rows,
        )

        fig.subplots_adjust(
            **PERFORMANCE_SPEC["layout"]
        )

        save_figure(
            fig,
            args.output_prefix.with_name(
                args.output_prefix.name
                + "_performance"
            ),
            png_dpi=args.png_dpi,
        )

        plt.close(fig)

    # -------------------------------------------------------------
    # Reliability figure
    # -------------------------------------------------------------
    if need_reliability:

        assert diagnostics is not None

        fig, ax = plt.subplots(
            figsize=(
                args.reliability_width,
                args.reliability_height,
            )
        )

        draw_reliability(
            ax,
            diagnostics,
        )

        fig.subplots_adjust(
            **RELIABILITY_SPEC["layout"]
        )

        save_figure(
            fig,
            args.output_prefix.with_name(
                args.output_prefix.name
                + "_vector_reliability"
            ),
            png_dpi=args.png_dpi,
        )

        plt.close(fig)

    # -------------------------------------------------------------
    # Summary CSV
    # -------------------------------------------------------------
    write_figure_summary(
        args.output_prefix.with_name(
            args.output_prefix.name
            + "_summary.csv"
        ),
        mc_rows,
        diagnostics,
    )

    # -------------------------------------------------------------
    # Output message
    # -------------------------------------------------------------
    if args.figure == "all":

        print(
            "Saved reference-scale separate figures:\n"
            f"  {args.output_prefix}_performance.[png|pdf|svg]\n"
            f"  {args.output_prefix}_vector_reliability.[png|pdf|svg]"
        )

    elif args.figure == "performance":

        print(
            f"Saved "
            f"{args.output_prefix}_performance.[png|pdf|svg]"
        )

    else:

        print(
            f"Saved "
            f"{args.output_prefix}_vector_reliability.[png|pdf|svg]"
        )


if __name__ == "__main__":
    main()