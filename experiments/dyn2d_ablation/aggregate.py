#!/usr/bin/env python3
"""Average Dyn2D grid metrics across independent folds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("MC1", "MC2", "MC3")


def parse_folds(text: str) -> list[int]:
    values = [int(x) for x in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique fold indices")
    return values


def load_fold(path: Path) -> tuple[int, dict[tuple[float, float], dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty fold CSV: {path}")
    folds = {int(float(row["fold"])) for row in rows}
    if len(folds) != 1:
        raise ValueError(f"{path} contains multiple folds: {sorted(folds)}")
    fold = next(iter(folds))
    by_cell: dict[tuple[float, float], dict[str, str]] = {}
    for row in rows:
        if not row.get("model_name"):
            raise ValueError(f"missing model_name in {path}")
        key = (float(row["strength"]), float(row["slerp_t"]))
        if key in by_cell:
            raise ValueError(f"duplicate cell {key} in {path}")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {metric} at {key} in {path}")
        by_cell[key] = row
    return fold, by_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--fold-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-folds", type=parse_folds, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    fold_tables: dict[int, dict[tuple[float, float], dict[str, str]]] = {}
    for path in args.fold_csvs:
        fold, table = load_fold(path)
        models = {row["model_name"] for row in table.values()}
        if models != {args.model}:
            raise ValueError(f"model mismatch in {path}: found {sorted(models)}, expected {args.model}")
        if fold in fold_tables:
            raise ValueError(f"duplicate input for fold {fold}")
        fold_tables[fold] = table
    if sorted(fold_tables) != sorted(args.expected_folds):
        raise ValueError(
            f"fold mismatch: found {sorted(fold_tables)}, expected {sorted(args.expected_folds)}"
        )

    reference_fold = args.expected_folds[0]
    reference_grid = set(fold_tables[reference_fold])
    for fold in args.expected_folds[1:]:
        if set(fold_tables[fold]) != reference_grid:
            missing = sorted(reference_grid - set(fold_tables[fold]))
            extra = sorted(set(fold_tables[fold]) - reference_grid)
            raise ValueError(f"grid mismatch for fold {fold}: missing={missing}, extra={extra}")

    records: list[dict] = []
    for strength, slerp_t in sorted(reference_grid, key=lambda x: (x[1], x[0])):
        record: dict[str, object] = {
            "model_name": args.model,
            "model_label": args.model_label,
            "strength": strength,
            "slerp_t": slerp_t,
        }
        fold_metrics: dict[str, dict[str, float]] = {}
        mean_metrics: dict[str, float] = {}
        for metric in METRICS:
            values = [float(fold_tables[fold][(strength, slerp_t)][metric]) for fold in args.expected_folds]
            for fold, value in zip(args.expected_folds, values):
                record[f"fold{fold}_{metric}"] = value
                fold_metrics.setdefault(str(fold), {})[metric] = value
            mean = float(np.mean(values))
            record[metric] = mean
            record[f"mean_{metric}"] = mean
            mean_metrics[metric] = mean
        record["fold_metrics"] = fold_metrics
        record["mean_metrics"] = mean_metrics
        records.append(record)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = ["model_name", "model_label", "strength", "slerp_t"]
    for metric in METRICS:
        csv_fields.extend([*(f"fold{fold}_{metric}" for fold in args.expected_folds), metric, f"mean_{metric}"])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    report = {
        "experiment": "Dyn2D strength x SLERP ablation",
        "aggregation": "arithmetic mean across folds",
        "model_name": args.model,
        "model_label": args.model_label,
        "folds": args.expected_folds,
        "cells": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fold-mean CSV: {args.output_csv}")
    print(f"Saved fold-mean JSON: {args.output_json}")


if __name__ == "__main__":
    main()
