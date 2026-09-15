#!/usr/bin/env python3
"""Plot validation-fraction ablations with the shared experiment style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plot_common import (  # noqa: E402
    EXPECTED_MODELS,
    finish,
    plot_mean_sd,
    save,
    setup_plot,
    write_csv,
)


def load_seed_series(input_dir: Path, seeds: list[int]) -> list[dict]:
    """Aggregate each model's MC1 over the requested random seeds."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Summary directory not found: {input_dir}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got {seeds}")

    payloads: dict[str, dict] = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("model", "results"):
            if key not in payload:
                raise ValueError(f"{path} is missing required field {key!r}")
        model = str(payload["model"])
        if model in payloads:
            raise ValueError(f"Duplicate summary for model={model}: {path}")
        if not isinstance(payload["results"], list) or not payload["results"]:
            raise ValueError(f"{path} contains no results")
        payloads[model] = payload

    missing_models = [model for model in EXPECTED_MODELS if model not in payloads]
    if missing_models:
        raise ValueError(
            "Missing model summaries: "
            + ", ".join(missing_models)
            + f". Expected files in {input_dir}"
        )

    series: list[dict] = []
    reference_x: np.ndarray | None = None
    for model in EXPECTED_MODELS:
        payload = payloads[model]
        grouped: dict[float, dict[int, float]] = {}
        for row in payload["results"]:
            try:
                x = float(row["val_fraction"])
                seed = int(row["seed"])
                value = float(row["MC1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{model} has an invalid val_fraction/seed/MC1 row: {row}"
                ) from exc
            if seed not in seeds:
                continue
            by_seed = grouped.setdefault(x, {})
            if seed in by_seed:
                raise ValueError(
                    f"Duplicate summary for model={model}, "
                    f"val_fraction={x:g}, seed={seed}"
                )
            by_seed[seed] = value

        if not grouped:
            raise ValueError(
                f"No requested seeds {seeds} found for model={model}"
            )

        x_values = np.asarray(sorted(grouped), dtype=float)
        values: list[list[float]] = []
        for x in x_values:
            missing_seeds = [seed for seed in seeds if seed not in grouped[x]]
            if missing_seeds:
                raise ValueError(
                    f"Missing seeds {missing_seeds} for model={model}, "
                    f"val_fraction={x:g}"
                )
            values.append([grouped[x][seed] for seed in seeds])
        fold_values = np.asarray(values, dtype=float).T

        if reference_x is None:
            reference_x = x_values
        elif len(x_values) != len(reference_x) or not np.allclose(
            x_values, reference_x, rtol=0, atol=1e-12
        ):
            raise ValueError("Llama and Qwen summaries use different x-axis grids")

        series.append(
            {
                "model": model,
                "label": str(payload.get("model_label") or model),
                "x": x_values,
                "fold_values": fold_values,
                "mean": fold_values.mean(axis=0),
                "sd": fold_values.std(axis=0, ddof=1)
                if len(seeds) > 1
                else np.zeros_like(x_values),
            }
        )

    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE.parents[1]
        / "results"
        / "experiment"
        / "val_fraction_ablation"
        / "summaries",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE.parents[1]
        / "results"
        / "experiment"
        / "val_fraction_ablation"
        / "val_fraction_ablation_mc1",
    )
    parser.add_argument("--seeds", default="42 123 0")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split()]

    setup_plot()
    series = load_seed_series(args.input_dir, seeds)
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    plot_mean_sd(axis, series)
    finish(axis, "Validation fraction")
    figure.tight_layout()
    save(figure, args.output_prefix)
    csv_path = write_csv(args.output_prefix, "val_fraction", series)
    plt.close(figure)
    print(f"Saved {args.output_prefix} and {csv_path}")


if __name__ == "__main__":
    main()
