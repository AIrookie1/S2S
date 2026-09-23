#!/usr/bin/env python3
"""Shared aggregation and plotting helpers for experiment figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from plot_specs import get_figure_size, get_figure_spec, is_visible, text_value
from plot_theme import COLORS, MODEL_STYLES, SAVE_SETTINGS, apply_style, get_style

EXPECTED_MODELS = (
    "llama3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
)


def setup_nature_style(profile_name: str = "quarter_quarter") -> dict[str, float]:
    """Backward-compatible entrypoint for the centralized Nature-style theme."""
    return apply_style(profile_name)


def setup_plot(figure_name: str | None = None, profile_name: str | None = None) -> dict[str, float]:
    """Apply the style assigned to a figure, or an explicit style profile."""
    if profile_name is None:
        profile_name = (
            str(get_figure_spec(figure_name)["style"])
            if figure_name is not None
            else "quarter_quarter"
        )
    return apply_style(profile_name)


def figure_size(figure_name: str) -> tuple[float, float]:
    return get_figure_size(figure_name)


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
    """Load per-model/per-fold summaries and aggregate MC1 as mean ± sample SD."""
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
                f"Duplicate summary for model={model}, fold={fold}: {previous} and {path}"
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
                f"Missing folds for {model}: {missing_folds}; expected folds {expected_folds}"
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
                raise ValueError(f"Inconsistent x grid for {model}, fold={fold}: {path}")
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

    reference_x = output[0]["x"]
    for series in output[1:]:
        if len(series["x"]) != len(reference_x) or not np.allclose(
            series["x"], reference_x, rtol=0, atol=1e-12
        ):
            raise ValueError("Llama and Qwen summaries use different x-axis grids")
    return output


def plot_model_series(
    ax,
    series_list: list[dict[str, Any]],
    *,
    figure_name: str | None = None,
    profile_name: str | None = None,
) -> None:
    """Overlay model means and fold-based error bars using central marker settings."""
    if profile_name is None:
        profile_name = (
            str(get_figure_spec(figure_name)["style"])
            if figure_name is not None
            else "quarter_quarter"
        )
    plot_style = get_style(profile_name)
    for series in series_list:
        model = series["model"]
        style = MODEL_STYLES.get(
            model,
            {
                "color": COLORS["dark_gray"],
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
            markeredgewidth=plot_style["marker_edge_width"],
            linewidth=plot_style["line_width"],
            markersize=plot_style["marker_size"],
            elinewidth=plot_style["errorbar_line_width"],
            capsize=plot_style["errorbar_cap_size"],
            capthick=plot_style["errorbar_cap_thickness"],
            zorder=3,
        )


def finish_axis(ax, *, figure_name: str | None = None) -> None:
    """Apply shared axes, grid and configurable legend treatment."""
    profile_name = (
        str(get_figure_spec(figure_name)["style"])
        if figure_name is not None
        else "quarter_quarter"
    )
    style = get_style(profile_name)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style["axes_line_width"])
    ax.spines["bottom"].set_linewidth(style["axes_line_width"])
    ax.tick_params(top=False, right=False, pad=2.5)
    ax.grid(axis="y", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)

    show_legend = True if figure_name is None else is_visible(figure_name, "legend", True)
    if show_legend:
        legend_kwargs = {
            "frameon": False,
            "loc": "best",
            "handlelength": 2.2,
            "handletextpad": 0.55,
            "borderaxespad": 0.25,
            "labelspacing": 0.35,
        }
        if figure_name is not None:
            legend_kwargs.update(get_figure_spec(figure_name).get("legend", {}))
        ax.legend(**legend_kwargs)
    ax.margins(x=0.035, y=0.12)


def apply_text(ax, figure_name: str, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply figure text and global show/hide switches from plot_specs.py."""
    xlabel = text_value(figure_name, "xlabel", xlabel) if xlabel is None else xlabel
    ylabel = text_value(figure_name, "ylabel", ylabel) if ylabel is None else ylabel
    title = text_value(figure_name, "title", None)
    if is_visible(figure_name, "xlabel", True) and xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
    if is_visible(figure_name, "ylabel", True) and ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    if is_visible(figure_name, "title", False) and title:
        ax.set_title(title)


def finish(ax, xlabel: str | None = None, *, figure_name: str | None = None) -> None:
    if figure_name is not None:
        apply_text(ax, figure_name, xlabel=xlabel)
    else:
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("MC1")
    finish_axis(ax, figure_name=figure_name)


def save_figure(figure, output_prefix: Path, *, png_dpi: int | None = None) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs: dict[str, Any] = {
            "facecolor": SAVE_SETTINGS["facecolor"],
            "bbox_inches": SAVE_SETTINGS["bbox_inches"],
            "pad_inches": SAVE_SETTINGS["pad_inches"],
        }
        if suffix == ".png":
            kwargs["dpi"] = int(png_dpi or SAVE_SETTINGS["png_dpi"])
        figure.savefig(output_prefix.with_suffix(suffix), **kwargs)


def save(fig, prefix: Path) -> None:
    save_figure(fig, prefix)


def write_aggregate_csv(
    output_prefix: Path,
    x_name: str,
    series_list: list[dict[str, Any]],
) -> Path:
    """Save exact replicate values, mean and SD used by the figure."""
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


def write_csv(prefix: Path, x_key: str, series_list):
    return write_aggregate_csv(prefix, x_key, series_list)


def plot_mean_sd(ax, series_list, *, figure_name: str | None = None) -> None:
    plot_model_series(ax, series_list, figure_name=figure_name)
