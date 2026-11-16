#!/usr/bin/env python3
"""Aggregate S2S_R MLP-hidden ablation metrics across folds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ("MC1", "MC2", "MC3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.num_folds < 1:
        raise ValueError("num_folds must be positive")
    expected_folds = list(range(args.num_folds))

    payloads: dict[str, dict[int, dict]] = {}
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", ""))
        if not model or "fold_idx" not in payload or not payload.get("results"):
            raise ValueError(f"invalid summary: {path}")
        fold = int(payload["fold_idx"])
        if fold in payloads.setdefault(model, {}):
            raise ValueError(f"duplicate model/fold summary: {model} fold={fold}")
        payloads[model][fold] = payload

    if not payloads:
        raise ValueError(f"no summary JSON files found in {args.input_dir}")

    records: list[dict] = []
    report_models: dict[str, dict] = {}
    for model in sorted(payloads):
        folds = payloads[model]
        missing = [fold for fold in expected_folds if fold not in folds]
        if missing:
            raise ValueError(f"missing folds for {model}: {missing}")
        extra = sorted(set(folds) - set(expected_folds))
        if extra:
            raise ValueError(f"unexpected folds for {model}: {extra}")

        fold_tables: dict[int, dict[int, dict]] = {}
        for fold in expected_folds:
            table: dict[int, dict] = {}
            for row in folds[fold]["results"]:
                hidden = int(row["mlp_hidden"])
                if hidden in table:
                    raise ValueError(f"duplicate hidden={hidden} for {model} fold={fold}")
                table[hidden] = row
            fold_tables[fold] = table

        hidden_ref = set(fold_tables[expected_folds[0]])
        for fold in expected_folds[1:]:
            if set(fold_tables[fold]) != hidden_ref:
                raise ValueError(f"hidden-size grid mismatch for {model} fold={fold}")

        model_label = str(folds[expected_folds[0]].get("model_label") or model)
        model_records: list[dict] = []
        for hidden in sorted(hidden_ref):
            record: dict[str, object] = {
                "model": model,
                "model_label": model_label,
                "mlp_hidden": hidden,
            }
            fold_metrics: dict[str, dict[str, float]] = {}
            mean_metrics: dict[str, float] = {}
            for metric in METRICS:
                values = [float(fold_tables[fold][hidden][metric]) for fold in expected_folds]
                if not all(np.isfinite(value) for value in values):
                    raise ValueError(f"non-finite {metric} for {model} hidden={hidden}")
                for fold, value in zip(expected_folds, values):
                    record[f"fold{fold}_{metric}"] = value
                    fold_metrics.setdefault(str(fold), {})[metric] = value
                mean = float(np.mean(values))
                record[f"mean_{metric}"] = mean
                mean_metrics[metric] = mean
            record["fold_metrics"] = fold_metrics
            record["mean_metrics"] = mean_metrics
            records.append(record)
            model_records.append(record)
        report_models[model] = {"model_label": model_label, "results": model_records}

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "model_label", "mlp_hidden"]
    for metric in METRICS:
        fields.extend([*(f"fold{fold}_{metric}" for fold in expected_folds), f"mean_{metric}"])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    report = {
        "experiment": "S2S_R MLP hidden-size ablation",
        "method": "S2S_R",
        "aggregation": "arithmetic mean across folds",
        "folds": expected_folds,
        "models": report_models,
    }
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved aggregate CSV: {args.output_csv}")
    print(f"Saved aggregate JSON: {args.output_json}")


if __name__ == "__main__":
    main()
