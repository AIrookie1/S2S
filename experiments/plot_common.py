#!/usr/bin/env python3
"""Shared two-model/two-fold aggregation and Nature-style plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np


EXPECTED_MODELS = (
    "llama3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
)

MODEL_STYLES: dict[str, dict[str, Any]] = {
    "llama3.1-8B-Instruct": {
        "label": "Llama 3.1 8B",
        "color": "#3C5488",
        "marker": "o",
        "linestyle": "-",
    },
    "Qwen2.5-7B-Instruct": {
        "label": "Qwen 2.5 7B",
        "color": "#E64B35",
        "marker": "s",
        "linestyle": "--",
    },
}


def setup_nature_style() -> None:
    """Compact, publication-ready styling inspired by Nature figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _validate_summary(payload: dict[str, Any], path: Path, x_key: str) -> None:
    required = ("model", "fold_idx", "results")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    if not isinstance(payload["results"], list) or not payload["results"]:
        raise ValueError(f"{path} contains no results")

    for row in payload["results"]:
        if x_key not in row or "MC1" not in row:
            raise ValueError(f"{path} has a result row without {x_key!r} or 'MC1'")


def load_fold_series(
    input_dir: Path,
    x_key: str,
    num_folds: int = 2,
) -> list[dict[str, Any]]:
    """
    Load independent per-model/per-fold summaries and aggregate MC1.

    Error bars are sample standard deviation across folds (ddof=1).
    For the requested setup, each point is therefore mean ± SD over fold0/fold1.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Summary directory not found: {input_dir}")

    payloads: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_summary(payload, path, x_key)
        model = str(payload["model"])
        fold = int(payload["fold_idx"])
        payloads.setdefault(model, {})
        if fold in payloads[model]:
            previous = payloads[model][fold][0]
            raise ValueError(
                f"Duplicate summary for model={model}, fold={fold}: "
                f"{previous} and {path}"
            )
        payloads[model][fold] = (path, payload)

    missing_models = [model for model in EXPECTED_MODELS if model not in payloads]
    if missing_models:
        raise ValueError(
            "Missing model summaries: "
            + ", ".join(missing_models)
            + f". Expected files in {input_dir}"
        )

    expected_folds = list(range(num_folds))
    output: list[dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        model_payloads = payloads[model]
        missing_folds = [fold for fold in expected_folds if fold not in model_payloads]
        if missing_folds:
            raise ValueError(
                f"Missing folds for {model}: {missing_folds}; "
                f"expected folds {expected_folds}"
            )

        x_ref: np.ndarray | None = None
        fold_values: list[np.ndarray] = []
        model_label: str | None = None

        for fold in expected_folds:
            path, payload = model_payloads[fold]
            rows = sorted(payload["results"], key=lambda row: float(row[x_key]))
            x = np.asarray([float(row[x_key]) for row in rows], dtype=float)
            y = np.asarray([float(row["MC1"]) for row in rows], dtype=float)

            if x_ref is None:
                x_ref = x
            elif len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Inconsistent x grid for {model}, fold={fold}: {path}"
                )

            fold_values.append(y)
            model_label = str(payload.get("model_label") or model_label or "")

        assert x_ref is not None
        values = np.vstack(fold_values)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if num_folds > 1 else np.zeros_like(mean)

        style = MODEL_STYLES.get(model, {})
        output.append(
            {
                "model": model,
                "label": model_label or style.get("label", model),
                "x": x_ref,
                "fold_values": values,
                "mean": mean,
                "sd": sd,
            }
        )

    # Enforce exactly the same ablation grid for the two models.
    reference_x = output[0]["x"]
    for series in output[1:]:
        if len(series["x"]) != len(reference_x) or not np.allclose(
            series["x"], reference_x, rtol=0, atol=1e-12
        ):
            raise ValueError("Llama and Qwen summaries use different x-axis grids")

    return output


def plot_model_series(ax, series_list: list[dict[str, Any]]) -> None:
    """Overlay model means and fold-based error bars."""
    for series in series_list:
        model = series["model"]
        style = MODEL_STYLES.get(
            model,
            {
                "color": "#4D4D4D",
                "marker": "o",
                "linestyle": "-",
                "label": series["label"],
            },
        )
        ax.errorbar(
            series["x"],
            series["mean"],
            yerr=series["sd"],
            label=series["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            linewidth=1.5,
            markersize=4.5,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )


def finish_axis(ax) -> None:
    """Apply the same axis/legend treatment to Fig. 3a, Fig. 4 and Fig. 5."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(top=False, right=False, pad=2.5)
    ax.grid(axis="y", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="best",
        handlelength=2.2,
        handletextpad=0.55,
        borderaxespad=0.25,
        labelspacing=0.35,
    )
    ax.margins(x=0.035, y=0.12)


def save_figure(figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        figure.savefig(output_prefix.with_suffix(suffix), **kwargs)


def write_aggregate_csv(
    output_prefix: Path,
    x_name: str,
    series_list: list[dict[str, Any]],
) -> Path:
    """Save the exact fold values, mean and SD used by the figure."""
    csv_path = output_prefix.with_name(output_prefix.name + "_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    num_folds = int(series_list[0]["fold_values"].shape[0])
    fieldnames = [
        "model",
        "model_label",
        x_name,
        *[f"fold{fold}_MC1" for fold in range(num_folds)],
        "mean_MC1",
        "sd_MC1",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for series in series_list:
            for idx, x in enumerate(series["x"]):
                row: dict[str, Any] = {
                    "model": series["model"],
                    "model_label": series["label"],
                    x_name: float(x),
                    "mean_MC1": float(series["mean"][idx]),
                    "sd_MC1": float(series["sd"][idx]),
                }
                for fold in range(num_folds):
                    row[f"fold{fold}_MC1"] = float(series["fold_values"][fold, idx])
                writer.writerow(row)
    return csv_path


# Backward-compatible API used by the experiment scripts restored from dyn6.
def setup_plot() -> None:
    setup_nature_style()

def plot_mean_sd(ax, series_list) -> None:
    plot_model_series(ax, series_list)

def finish(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MC1")
    finish_axis(ax)

def save(fig, prefix: Path) -> None:
    save_figure(fig, prefix)

def write_csv(prefix: Path, x_key: str, series_list):
    return write_aggregate_csv(prefix, x_key, series_list)
