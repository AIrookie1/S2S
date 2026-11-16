#!/usr/bin/env python3
"""Plot residual-direction analysis as three independent publication figures."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import save
from plot_specs import get_figure_size, get_figure_spec
from plot_theme import apply_style, get_style

FIGURE_NAME = "fig4ab_residual_transfer"
SPEC = get_figure_spec(FIGURE_NAME)
CONDITION_ORDER = (
    "mean_only",
    "shuffled_residual",
    "cross_category_cosine_nearest_residual",
    "predicted_residual",
    "oracle_residual",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot residual analysis")
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./results/experiment/fig4ab_residual_transfer"),
    )
    p.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help="Base prefix. Outputs append _alignment, _swap, and _geometric_gain.",
    )
    p.add_argument(
        "--allow_missing_mc",
        action="store_true",
        help="Plot Panels (a,c) even when --skip_mc was used.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty result file: {path}")
    return rows


def clean_axis(ax) -> None:
    style = get_style(str(SPEC["style"]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style["axes_line_width"])
    ax.spines["bottom"].set_linewidth(style["axes_line_width"])
    ax.tick_params(top=False, right=False, pad=2.5)
    ax.grid(axis="y", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)


def draw_panel_a(ax, rows: list[dict[str, str]]) -> None:
    columns = (
        "matched",
        "within_category_shuffled",
        "global_shuffled",
        "cross_category_cosine_nearest",
    )
    labels = (
        "Matched",
        "Category\nshuffled",
        "Global\nshuffled",
        "Cross-category\ncosine-nearest",
    )
    values = []
    for column in columns:
        arr = np.asarray([float(row[column]) for row in rows], dtype=float)
        values.append(arr[np.isfinite(arr)])

    rng = np.random.default_rng(0)
    for pos, arr in enumerate(values, start=1):
        if len(arr) >= 2:
            parts = ax.violinplot(
                [arr],
                positions=[pos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_alpha(0.24)
        jitter = rng.normal(0.0, 0.035, size=len(arr))
        ax.scatter(
            np.full(len(arr), pos) + jitter,
            arr,
            s=18,
            alpha=0.18,
            linewidths=0,
            zorder=4,
        )
        if len(arr):
            q25, median, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
            ax.vlines(pos, q25, q75, linewidth=5.0, zorder=4)
            ax.scatter([pos], [median], s=28, facecolors="white", edgecolors="black", linewidths=0.9, zorder=5)
        else:
            ax.text(pos, 0.02, "n/a", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.0, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel(SPEC["text"]["panel_a_ylabel"])
    ax.set_title("Residual alignment", loc="left", fontweight="bold")
    clean_axis(ax)


def draw_panel_b(ax, rows: list[dict[str, str]] | None, *, mean_label: str, ylabel: str) -> None:
    ax.axhline(0.0, linestyle=":", linewidth=0.9, alpha=0.6, zorder=0)
    if rows is None:
        ax.text(
            0.5,
            0.5,
            "MC residual-swap evaluation not run",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title("Residual swap", loc="left", fontweight="bold")
        clean_axis(ax)
        return

    by_condition = {row["condition"]: row for row in rows}
    selected = [by_condition[name] for name in CONDITION_ORDER]
    x = np.arange(len(selected))
    y = np.asarray([float(row["delta_MC1_pp_over_mean_only"]) for row in selected])
    lo = np.asarray([float(row["delta_MC1_pp_ci_low"]) for row in selected])
    hi = np.asarray([float(row["delta_MC1_pp_ci_high"]) for row in selected])
    yerr = np.vstack([y - lo, hi - y])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=10.2,
        linewidth=1.35,
        elinewidth=1.1,
        capsize=3.0,
        zorder=4,
    )
    ax.set_xticks(
        x,
        [
            mean_label,
            "Shuffled\nresidual",
            "Cross-category\ncosine-nearest",
            "Predicted\nresidual",
            "Oracle\nresidual",
        ],
    )
    ax.set_ylabel(ylabel)
    ax.set_title("Residual swap", loc="left", fontweight="bold")
    clean_axis(ax)


def draw_panel_c(ax, rows: list[dict[str, str]], *, ylabel: str) -> None:
    rows = sorted(rows, key=lambda row: int(row["quintile"]))
    x = np.asarray([int(row["quintile"]) for row in rows])
    y = np.asarray([float(row["mean_angular_gain_deg"]) for row in rows])
    lo = np.asarray([float(row["ci_low"]) for row in rows])
    hi = np.asarray([float(row["ci_high"]) for row in rows])
    yerr = np.vstack([y - lo, hi - y])

    ax.axhline(0.0, linestyle=":", linewidth=0.9, alpha=0.6, zorder=0)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.35,
        markersize=5.2,
        elinewidth=1.1,
        capsize=3.0,
        zorder=4,
    )
    ax.set_xticks(x, [f"Q{i}" for i in x])
    ax.set_xlabel(SPEC["text"]["panel_c_xlabel"])
    ax.set_ylabel(ylabel)
    ax.set_title("Residual quality vs. geometric gain", loc="left", fontweight="bold")
    clean_axis(ax)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_prefix = args.output_prefix or (input_dir / "fig4ab_residual_transfer")

    alignment_rows = read_csv(input_dir / "alignment_question.csv")
    quintile_rows = read_csv(input_dir / "quintile_summary.csv")
    mc_path = input_dir / "mc_summary.csv"
    if mc_path.is_file():
        mc_rows = read_csv(mc_path)
    elif args.allow_missing_mc:
        mc_rows = None
    else:
        raise FileNotFoundError(
            f"{mc_path} not found. Run evaluate.py without --skip_mc or pass --allow_missing_mc."
        )

    summary_path = input_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    steering_method = summary.get("steering_method")
    if steering_method == "S2S_L_Static":
        mean_label = "Mean only\n(Linear)"
        panel_b_ylabel = r"$\Delta$ MC1 over Linear mean-only (pp)"
        panel_c_ylabel = "Angular-error reduction vs. Linear mean (deg)"
    elif steering_method == "S2S_R":
        mean_label = "Mean only\n(Spherical)"
        panel_b_ylabel = r"$\Delta$ MC1 over Spherical mean-only (pp)"
        panel_c_ylabel = "Angular-error reduction vs. Spherical (deg)"
    else:
        mean_label = "Mean only"
        panel_b_ylabel = SPEC["text"]["panel_b_ylabel"]
        panel_c_ylabel = SPEC["text"]["panel_c_ylabel"]

    apply_style(str(SPEC["style"]))
    figure_size = get_figure_size(FIGURE_NAME)
    layout = SPEC.get("layout", {})

    figures = (
        ("alignment", lambda ax: draw_panel_a(ax, alignment_rows)),
        ("swap", lambda ax: draw_panel_b(ax, mc_rows, mean_label=mean_label, ylabel=panel_b_ylabel)),
        ("geometric_gain", lambda ax: draw_panel_c(ax, quintile_rows, ylabel=panel_c_ylabel)),
    )
    for suffix, draw in figures:
        fig, ax = plt.subplots(figsize=figure_size)
        draw(ax)
        fig.subplots_adjust(**layout)
        save(fig, Path(f"{output_prefix}_{suffix}"))
        plt.close(fig)

    if summary:
        print(
            "Residual analysis: "
            f"method={summary.get('steering_method')} "
            f"model={summary.get('model_name')} layer={summary.get('layer')} "
            f"fold={summary.get('fold_idx')}"
        )
    print(f"Saved {output_prefix}_alignment/swap/geometric_gain.png/.pdf")


if __name__ == "__main__":
    main()
