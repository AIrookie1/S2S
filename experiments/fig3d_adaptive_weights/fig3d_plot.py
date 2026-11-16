#!/usr/bin/env python3
"""Nature-style ternary density plots for TruthfulQA adaptive weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

formatter = FuncFormatter(lambda x, pos: f"{x:g}")


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


# ---------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------

from plot_common import figure_size, save_figure, setup_plot  # noqa: E402
from plot_specs import get_figure_spec  # noqa: E402
from plot_theme import COLORS  # noqa: E402

FIGURE_NAME = "fig3d_adaptive_weights"
FIGURE_SPEC = get_figure_spec(FIGURE_NAME)


# ---------------------------------------------------------------------
# Ternary geometry
# ---------------------------------------------------------------------

VERTEX_ALPHA = np.array(
    [0.5, math.sqrt(3.0) / 2.0],
    dtype=float,
)

VERTEX_BETA = np.array(
    [0.0, 0.0],
    dtype=float,
)

VERTEX_GAMMA = np.array(
    [1.0, 0.0],
    dtype=float,
)

TRIANGLE_VERTICES = np.vstack(
    [
        VERTEX_ALPHA,
        VERTEX_BETA,
        VERTEX_GAMMA,
    ]
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "figures",
    )
    parser.add_argument(
        "--model-label",
        default=None,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_kappa(value: float) -> str:
    """Format kappa cleanly for output filenames."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def ternary_xy(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    gamma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ternary coordinates (alpha, beta, gamma) satisfying
    alpha + beta + gamma = 1 to Cartesian coordinates.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)

    points = (
        alpha_arr[..., None] * VERTEX_ALPHA
        + beta_arr[..., None] * VERTEX_BETA
        + gamma_arr[..., None] * VERTEX_GAMMA
    )
    return points[..., 0], points[..., 1]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_density_files(input_dir: Path) -> list[dict[str, Any]]:
    """Load Fig. 7 adaptive-weight density JSON files."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Fig. 7 input directory not found: {input_dir}"
        )

    records: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("experiment") != "fig7_adaptive_weight_density":
            continue

        required = (
            "weight_kappa",
            "bins",
            "histogram",
            "mean_adaptive_weights",
        )
        missing = [key for key in required if key not in payload]

        if missing:
            raise ValueError(
                f"{path} is missing Fig. 7 fields: {missing}"
            )

        bins = int(payload["bins"])
        histogram = np.asarray(
            payload["histogram"],
            dtype=float,
        )

        if histogram.shape != (bins, bins):
            raise ValueError(
                f"{path} histogram shape "
                f"{histogram.shape} != ({bins}, {bins})"
            )

        mean = payload["mean_adaptive_weights"]
        mean_values = np.asarray(
            [
                mean["alpha"],
                mean["beta"],
                mean["gamma"],
            ],
            dtype=float,
        )

        if not np.isfinite(mean_values).all():
            raise ValueError(
                f"{path} has non-finite mean adaptive weights: "
                f"{mean_values}"
            )

        if not np.isclose(
            mean_values.sum(),
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                f"{path} has invalid mean adaptive weights: "
                f"{mean_values}"
            )

        records.append(
            {
                "path": path,
                "payload": payload,
                "histogram": histogram,
                "mean": mean_values,
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No Fig. 7 density JSON files found in {input_dir}"
        )

    records.sort(
        key=lambda item: float(item["payload"]["weight_kappa"])
    )
    return records


# ---------------------------------------------------------------------
# Density calculation
# ---------------------------------------------------------------------

def density_points(
    histogram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the beta/gamma histogram into ternary Cartesian positions
    and log-scaled token density values.
    """
    bins = histogram.shape[0]

    beta_values: list[float] = []
    gamma_values: list[float] = []
    values: list[float] = []

    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("Adaptive-weight histogram is empty")

    for beta_idx in range(bins):
        for gamma_idx in range(bins):
            # Only keep cells satisfying beta + gamma < 1.
            if beta_idx + gamma_idx >= bins:
                continue

            count = float(histogram[beta_idx, gamma_idx])
            beta = (beta_idx + 0.5) / bins
            gamma = (gamma_idx + 0.5) / bins

            beta_values.append(beta)
            gamma_values.append(gamma)

            # Tokens per million followed by log1p.
            values.append(
                np.log1p(1.0e6 * count / total)
            )

    beta_arr = np.asarray(beta_values, dtype=float)
    gamma_arr = np.asarray(gamma_values, dtype=float)
    alpha_arr = 1.0 - beta_arr - gamma_arr

    x, y = ternary_xy(
        alpha_arr,
        beta_arr,
        gamma_arr,
    )

    return (
        x,
        y,
        np.asarray(values, dtype=float),
    )


# ---------------------------------------------------------------------
# Ternary grid
# ---------------------------------------------------------------------

def add_ternary_grid(ax) -> None:
    """
    Add alpha/beta/gamma isolines.

    No labels are drawn because Fig. 7 is intended to contain
    no textual elements.
    """
    levels = tuple(FIGURE_SPEC["ternary"]["grid_levels"])

    vertex_sets = (
        (VERTEX_ALPHA, VERTEX_BETA, VERTEX_GAMMA),
        (VERTEX_BETA, VERTEX_ALPHA, VERTEX_GAMMA),
        (VERTEX_GAMMA, VERTEX_ALPHA, VERTEX_BETA),
    )

    for value in levels:
        for vertex, other_a, other_b in vertex_sets:
            p0 = value * vertex + (1.0 - value) * other_a
            p1 = value * vertex + (1.0 - value) * other_b

            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=COLORS["white"],
                linewidth=FIGURE_SPEC["ternary"]["grid_line_width"],
                alpha=FIGURE_SPEC["ternary"]["grid_alpha"],
                zorder=3,
            )


# ---------------------------------------------------------------------
# Plot one kappa
# ---------------------------------------------------------------------

def plot_one(
    record: dict[str, Any],
    output_dir: Path,
    vmax: float,
) -> Path:
    """Generate one text-free ternary density figure."""
    payload = record["payload"]
    kappa = float(payload["weight_kappa"])

    x, y, values = density_points(
        record["histogram"]
    )

    # -----------------------------------------------------------------
    # Triangulation
    # -----------------------------------------------------------------

    triangulation = mtri.Triangulation(
        x,
        y,
    )

    triangles = triangulation.triangles
    centroids = np.column_stack(
        (
            np.mean(x[triangles], axis=1),
            np.mean(y[triangles], axis=1),
        )
    )

    sqrt3 = math.sqrt(3.0)

    inside = (
        (centroids[:, 1] >= -1e-8)
        & (
            centroids[:, 1]
            <= sqrt3 / 2.0 + 1e-8
        )
        & (
            centroids[:, 1]
            <= sqrt3 * centroids[:, 0] + 1e-8
        )
        & (
            centroids[:, 1]
            <= sqrt3 * (1.0 - centroids[:, 0]) + 1e-8
        )
    )

    triangulation.set_mask(~inside)

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figure_size(FIGURE_NAME))

    # Reference palette:
    # low density: dark purple
    # medium density: blue/green
    # high density: yellow
    cmap = plt.get_cmap("viridis")

    density_max = max(
        vmax,
        1e-6,
    )

    ax.tricontourf(
        triangulation,
        values,
        levels=np.linspace(
            0.0,
            density_max,
            32,
        ),
        cmap=cmap,
        norm=Normalize(
            vmin=0.0,
            vmax=density_max,
        ),
        antialiased=True,
        extend="max",
        zorder=1,
    )

    # -----------------------------------------------------------------
    # Ternary grid
    # -----------------------------------------------------------------

    add_ternary_grid(ax)

    # -----------------------------------------------------------------
    # Triangle boundary
    # -----------------------------------------------------------------

    ax.plot(
        [
            VERTEX_ALPHA[0],
            VERTEX_BETA[0],
            VERTEX_GAMMA[0],
            VERTEX_ALPHA[0],
        ],
        [
            VERTEX_ALPHA[1],
            VERTEX_BETA[1],
            VERTEX_GAMMA[1],
            VERTEX_ALPHA[1],
        ],
        color=COLORS["black"],
        linewidth=FIGURE_SPEC["ternary"]["boundary_line_width"],
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )

    # -----------------------------------------------------------------
    # Mean adaptive-weight marker
    # -----------------------------------------------------------------

    mean = record["mean"]
    mean_x, mean_y = ternary_xy(
        mean[0],
        mean[1],
        mean[2],
    )

    ax.scatter(
        [float(mean_x)],
        [float(mean_y)],
        marker=FIGURE_SPEC["ternary"]["mean_marker"],
        s=FIGURE_SPEC["ternary"]["mean_marker_size"],
        color=COLORS["black"],
        edgecolor=COLORS["white"],
        linewidth=FIGURE_SPEC["ternary"]["mean_marker_line_width"],
        zorder=10,
    )

    # -----------------------------------------------------------------
    # Layout
    #
    # Intentionally:
    # - no title
    # - no alpha/beta/gamma labels
    # - no axes
    # - no tick labels
    # - no legend
    # - no annotation
    # - no colorbar
    # -----------------------------------------------------------------

    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_xlim(*FIGURE_SPEC["ternary"]["xlim"])
    ylim_pad = FIGURE_SPEC["ternary"]["ylim_pad"]
    ax.set_ylim(
        -ylim_pad,
        math.sqrt(3.0) / 2.0 + ylim_pad
    )
    ax.axis("off")

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            1.0,
        ),
        pad=FIGURE_SPEC["ternary"]["tight_pad"],
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    kappa_label = format_kappa(
        kappa
    ).replace(
        ".",
        "p",
    )

    output_prefix = (
        output_dir
        / f"fig7_adaptive_weights_kappa{kappa_label}"
    )

    # Use the shared Nature-style save helper.
    save_figure(
        fig,
        output_prefix,
    )

    plt.close(fig)

    return output_prefix.with_suffix(
        ".png"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate all Fig. 7 ternary density figures and a summary manifest."""
    args = parse_args()

    # Apply the same centralized experiment style as the other figures.
    setup_plot(FIGURE_NAME)

    records = load_density_files(
        args.input_dir
    )

    # Only plot kappa = 1 and 10
    target_kappas = {1.0, 3.0, 5.0, 10.0, 20.0}
    records = [
        item for item in records
        if float(item["payload"]["weight_kappa"]) in target_kappas
    ]

    # One shared density scale across every kappa figure.
    vmax = max(
        float(
            np.max(
                density_points(
                    item["histogram"]
                )[2]
            )
        )
        for item in records
    )

    # Kept for the manifest only.
    model_label = (
        args.model_label
        or str(
            records[0]["payload"].get("model_label")
            or records[0]["payload"].get("model_name")
            or "TruthfulQA"
        )
    )

    outputs = [
        str(
            plot_one(
                item,
                args.output_dir,
                vmax,
            )
        )
        for item in records
    ]

    # -----------------------------------------------------------------
    # Metadata manifest
    # -----------------------------------------------------------------

    manifest = {
        "experiment": "fig7_adaptive_weight_density",
        "model_label": model_label,
        "density_transform": "log1p(tokens / total_tokens * 1e6)",
        "vertex_order": [
            "alpha",
            "beta",
            "gamma",
        ],
        "figure_text": False,
        "colorbar": False,
        "shared_density_scale": True,
        "weights_gate_modes": sorted({
            str(item["payload"].get("weights_gate_mode", "all"))
            for item in records
        }),
        "files": [
            {
                "kappa": float(
                    item["payload"]["weight_kappa"]
                ),
                "input": str(
                    item["path"]
                ),
                "figure": output,
                "mean_adaptive_weights": (
                    item["mean"].tolist()
                ),
                "token_count": int(
                    item["payload"].get("token_count", 0)
                ),
                "weights_gate_mode": str(
                    item["payload"].get("weights_gate_mode", "all")
                ),
            }
            for item, output in zip(
                records,
                outputs,
            )
        ],
    }

    summary_path = (
        args.output_dir
        / "fig7_density_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(outputs)} "
        f"ternary density figures in "
        f"{args.output_dir}"
    )

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()