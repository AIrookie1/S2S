#!/usr/bin/env python3
"""Nature-style ternary density plots for TruthfulQA adaptive weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon

import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from plot_common1 import setup_nature_style  # noqa: E402


VERTEX_ALPHA = np.array([0.5, math.sqrt(3.0) / 2.0])
VERTEX_BETA = np.array([0.0, 0.0])
VERTEX_GAMMA = np.array([1.0, 0.0])
TRIANGLE_VERTICES = np.vstack([VERTEX_ALPHA, VERTEX_BETA, VERTEX_GAMMA])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--model-label", default=None)
    return parser.parse_args()


def format_kappa(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def ternary_xy(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = (
        alpha[..., None] * VERTEX_ALPHA
        + beta[..., None] * VERTEX_BETA
        + gamma[..., None] * VERTEX_GAMMA
    )
    return points[..., 0], points[..., 1]


def load_density_files(input_dir: Path) -> list[dict[str, Any]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Fig. 7 input directory not found: {input_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "fig7_adaptive_weight_density":
            continue
        required = ("weight_kappa", "bins", "histogram", "mean_adaptive_weights")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing Fig. 7 fields: {missing}")
        bins = int(payload["bins"])
        histogram = np.asarray(payload["histogram"], dtype=float)
        if histogram.shape != (bins, bins):
            raise ValueError(f"{path} histogram shape {histogram.shape} != ({bins}, {bins})")
        mean = payload["mean_adaptive_weights"]
        mean_values = np.asarray([mean["alpha"], mean["beta"], mean["gamma"]], dtype=float)
        if not np.isfinite(mean_values).all() or not np.isclose(mean_values.sum(), 1.0, atol=1e-4):
            raise ValueError(f"{path} has invalid mean adaptive weights: {mean_values}")
        records.append({"path": path, "payload": payload, "histogram": histogram, "mean": mean_values})
    if not records:
        raise FileNotFoundError(f"No Fig. 7 density JSON files found in {input_dir}")
    records.sort(key=lambda item: float(item["payload"]["weight_kappa"]))
    return records


def density_points(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = histogram.shape[0]
    beta_values: list[float] = []
    gamma_values: list[float] = []
    values: list[float] = []
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("Adaptive-weight histogram is empty")
    for beta_idx in range(bins):
        for gamma_idx in range(bins):
            if beta_idx + gamma_idx >= bins:
                continue
            count = float(histogram[beta_idx, gamma_idx])
            beta_values.append((beta_idx + 0.5) / bins)
            gamma_values.append((gamma_idx + 0.5) / bins)
            values.append(np.log1p(1.0e6 * count / total))
    alpha = 1.0 - np.asarray(beta_values) - np.asarray(gamma_values)
    x, y = ternary_xy(alpha, np.asarray(beta_values), np.asarray(gamma_values))
    return x, y, np.asarray(values)


def add_ternary_grid(ax) -> None:
    levels = (0.2, 0.4, 0.6, 0.8)
    for value in levels:
        # Constant alpha, beta and gamma isolines.
        for vertex, other_a, other_b in (
            (VERTEX_ALPHA, VERTEX_BETA, VERTEX_GAMMA),
            (VERTEX_BETA, VERTEX_ALPHA, VERTEX_GAMMA),
            (VERTEX_GAMMA, VERTEX_ALPHA, VERTEX_BETA),
        ):
            p0 = value * vertex + (1.0 - value) * other_a
            p1 = value * vertex + (1.0 - value) * other_b
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="#FFFFFF",
                linewidth=0.45,
                alpha=0.55,
                zorder=3,
            )


def plot_one(record: dict[str, Any], output_dir: Path, vmax: float, model_label: str) -> Path:
    payload = record["payload"]
    kappa = float(payload["weight_kappa"])
    x, y, values = density_points(record["histogram"])
    triangulation = mtri.Triangulation(x, y)
    centroids = np.column_stack(
        (
            np.mean(x[triangulation.triangles], axis=1),
            np.mean(y[triangulation.triangles], axis=1),
        )
    )
    inside = (
        (centroids[:, 1] >= -1e-8)
        & (centroids[:, 1] <= math.sqrt(3.0) / 2.0 + 1e-8)
        & (centroids[:, 1] <= math.sqrt(3.0) * centroids[:, 0] + 1e-8)
        & (centroids[:, 1] <= math.sqrt(3.0) * (1.0 - centroids[:, 0]) + 1e-8)
    )
    triangulation.set_mask(~inside)

    fig, ax = plt.subplots(figsize=(5.25, 4.95))
    cmap = LinearSegmentedColormap.from_list(
        "nature_density",
        ["#F7FBFF", "#D7E6F5", "#86A8CE", "#3C5488", "#E64B35"],
        N=256,
    )
    contour = ax.tricontourf(
        triangulation,
        values,
        levels=np.linspace(0.0, max(vmax, 1e-6), 32),
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=max(vmax, 1e-6)),
        antialiased=True,
        extend="max",
        zorder=1,
    )
    triangle_patch = Polygon(TRIANGLE_VERTICES, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(triangle_patch)
    # The triangulation is masked by triangle-centroid geometry above, so no
    # additional clipping collection is needed across Matplotlib versions.

    add_ternary_grid(ax)
    ax.plot(
        [*TRIANGLE_VERTICES[:, 0], TRIANGLE_VERTICES[0, 0]],
        [*TRIANGLE_VERTICES[:, 1], TRIANGLE_VERTICES[0, 1]],
        color="#222222",
        linewidth=1.5,
        zorder=5,
    )

    mean = record["mean"]
    mean_x, mean_y = ternary_xy(mean[0], mean[1], mean[2])
    ax.scatter([mean_x], [mean_y], marker="*", s=95, color="#222222", edgecolor="white", linewidth=0.8, zorder=7)
    ax.text(
        0.98,
        0.03,
        f"mean: ({mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=8,
    )

    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.065, r"$\alpha=1$", ha="center", va="bottom", fontsize=16)
    ax.text(-0.035, -0.045, r"$\beta=1$", ha="left", va="top", fontsize=16)
    ax.text(1.035, -0.045, r"$\gamma=1$", ha="right", va="top", fontsize=16)
    ax.set_title(f"TruthfulQA adaptive weights · $\\kappa={format_kappa(kappa)}$", pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, math.sqrt(3.0) / 2.0 + 0.12)
    ax.axis("off")

    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.035, shrink=0.86)
    colorbar.set_label(r"log$_{e}$(tokens per million + 1)", fontsize=11.5)
    colorbar.ax.tick_params(labelsize=10)
    fig.text(
        0.5,
        0.015,
        f"{model_label} · token density over " + r"$\alpha+\beta+\gamma=1$",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0), pad=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"fig7_adaptive_weights_kappa{format_kappa(kappa).replace('.', 'p')}"
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(prefix.with_suffix(suffix), facecolor="white", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)
    return prefix.with_suffix(".png")


def main() -> None:
    args = parse_args()
    setup_nature_style()
    records = load_density_files(args.input_dir)
    vmax = max(float(np.max(density_points(item["histogram"])[2])) for item in records)
    label = args.model_label or str(records[0]["payload"].get("model_label") or records[0]["payload"].get("model_name") or "TruthfulQA")
    outputs = [str(plot_one(item, args.output_dir, vmax, label)) for item in records]

    manifest = {
        "experiment": "fig7_adaptive_weight_density",
        "model_label": label,
        "density_transform": "log1p(tokens / total_tokens * 1e6)",
        "vertex_order": ["alpha", "beta", "gamma"],
        "files": [
            {
                "kappa": float(item["payload"]["weight_kappa"]),
                "input": str(item["path"]),
                "figure": output,
                "mean_adaptive_weights": item["mean"].tolist(),
                "token_count": int(item["payload"].get("token_count", 0)),
            }
            for item, output in zip(records, outputs)
        ],
    }
    (args.output_dir / "fig7_density_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(outputs)} ternary density figures in {args.output_dir}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
