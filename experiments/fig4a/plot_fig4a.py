#!/usr/bin/env python3
"""Fig. 4a: validation-fraction sweep on fold0, repeated across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from plot_common1 import (  # noqa: E402
    EXPECTED_MODELS,
    finish_axis,
    plot_model_series,
    save_figure,
    setup_nature_style,
)


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item) for item in value.split() if item]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide at least two distinct seeds")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 4a with seed-based error bars on fold0."
    )
    parser.add_argument("--input-dir", type=Path, default=HERE / "summaries")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "fig4a_val_fraction_mc1",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=[42, 123],
        help="space-separated random seeds used for each validation fraction",
    )
    return parser.parse_args()


def load_seed_series(input_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Summary directory not found: {input_dir}")

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", ""))
        if model in summaries:
            raise ValueError(f"Duplicate model summary: {summaries[model]['path']} and {path}")
        if model not in EXPECTED_MODELS:
            continue
        if int(payload.get("fold_idx", -1)) != 0:
            raise ValueError(f"Fig. 4a requires fold_idx=0 in {path}")
        if not isinstance(payload.get("results"), list) or not payload["results"]:
            raise ValueError(f"{path} contains no results")
        summaries[model] = {"path": path, "payload": payload}

    missing_models = [model for model in EXPECTED_MODELS if model not in summaries]
    if missing_models:
        raise ValueError("Missing model summaries: " + ", ".join(missing_models))

    output: list[dict[str, Any]] = []
    reference_x: np.ndarray | None = None
    for model in EXPECTED_MODELS:
        path = summaries[model]["path"]
        payload = summaries[model]["payload"]
        rows = payload["results"]
        grouped: dict[tuple[float, int], float] = {}
        x_values: set[float] = set()
        for row in rows:
            x = float(row["val_fraction"])
            seed = int(row["seed"])
            if seed not in seeds:
                continue
            key = (x, seed)
            if key in grouped:
                raise ValueError(f"Duplicate val_fraction/seed={key} in {path}")
            grouped[key] = float(row["MC1"])
            x_values.add(x)

        if not x_values:
            raise ValueError(f"No requested seeds {seeds} found in {path}")
        x = np.asarray(sorted(x_values), dtype=float)
        values = np.empty((len(seeds), len(x)), dtype=float)
        for seed_idx, seed in enumerate(seeds):
            for x_idx, fraction in enumerate(x):
                key = (float(fraction), seed)
                if key not in grouped:
                    raise ValueError(f"Missing val_fraction={fraction}, seed={seed} in {path}")
                values[seed_idx, x_idx] = grouped[key]

        if reference_x is None:
            reference_x = x
        elif len(x) != len(reference_x) or not np.allclose(x, reference_x, rtol=0, atol=1e-12):
            raise ValueError("Llama and Qwen summaries use different x-axis grids")

        style = payload.get("model_label")
        output.append(
            {
                "model": model,
                "label": str(style or model),
                "x": x,
                # plot_model_series uses the historical fold_values key; here
                # its rows are the independent random seeds.
                "fold_values": values,
                "mean": values.mean(axis=0),
                "sd": values.std(axis=0, ddof=1),
            }
        )

    return output


def write_seed_csv(output_prefix: Path, seeds: list[int], series_list: list[dict[str, Any]]) -> Path:
    csv_path = output_prefix.with_name(output_prefix.name + "_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "model_label",
        "val_fraction",
        *[f"seed{seed}_MC1" for seed in seeds],
        "mean_MC1",
        "sd_MC1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series in series_list:
            for idx, fraction in enumerate(series["x"]):
                row: dict[str, Any] = {
                    "model": series["model"],
                    "model_label": series["label"],
                    "val_fraction": float(fraction),
                    "mean_MC1": float(series["mean"][idx]),
                    "sd_MC1": float(series["sd"][idx]),
                }
                for seed_idx, seed in enumerate(seeds):
                    row[f"seed{seed}_MC1"] = float(series["fold_values"][seed_idx, idx])
                writer.writerow(row)
    return csv_path


def main() -> None:
    args = parse_args()
    setup_nature_style()
    series = load_seed_series(args.input_dir, args.seeds)

    fig, ax = plt.subplots(figsize=(4.9, 4.0))
    plot_model_series(ax, series)

    x = series[0]["x"]
    ax.set_xlabel("Validation fraction")
    ax.set_ylabel("MC1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.1f}" for value in x])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    finish_axis(ax)

    fig.tight_layout(pad=0.45)
    save_figure(fig, args.output_prefix)
    csv_path = write_seed_csv(args.output_prefix, args.seeds, series)
    plt.close(fig)

    seed_text = ", ".join(str(seed) for seed in args.seeds)
    print(f"Error bars: mean +/- sample SD across seeds [{seed_text}], fold0 only")
    print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
    print(f"Saved plotted data: {csv_path}")


if __name__ == "__main__":
    main()
