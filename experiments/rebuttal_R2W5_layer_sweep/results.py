"""Pure-stdlib cache and fold-mean selection utilities for the layer sweep."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

SELECTION = "test_set_sweep_best"
METRICS = ("MC1", "MC2", "MC3")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def valid_cell(report, context, method, parameters):
    if (report.get("context") != context or report.get("method") != method
            or report.get("parameters") != parameters):
        return False
    try:
        return (report["selection"] == SELECTION and report["n_questions"] > 0
                and len(report["test_q_indices"]) == report["n_questions"]
                and all(math.isfinite(report["metrics"][m])
                        and 0 <= report["metrics"][m] <= 1 for m in METRICS))
    except (KeyError, TypeError, ValueError):
        return False


def aggregate(reports):
    """Select one shared parameter combination AFTER averaging the two folds."""
    grouped = {}
    fold_ids = {}
    for report in reports:
        context = report["context"]
        method, layer, fold = report["method"], context["layer"], context["fold"]
        if not valid_cell(report, context, method, report["parameters"]):
            raise ValueError("Invalid evaluation cell")
        if fold not in (0, 1):
            raise ValueError("Expected folds 0 and 1")
        ids = tuple(report["test_q_indices"])
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate test question IDs")
        # Identical held-out questions across methods, hyperparameters and layers.
        if fold in fold_ids and fold_ids[fold] != ids:
            raise ValueError("Held-out question IDs differ across sweep cells")
        fold_ids[fold] = ids
        params = tuple(sorted(report["parameters"].items()))
        key = (method, layer, params)
        group = grouped.setdefault(key, {})
        if fold in group:
            raise ValueError(f"Duplicate fold for {key}")
        group[fold] = report
    if not reports or set(fold_ids) != {0, 1}:
        raise ValueError("Both folds are required")
    if set(fold_ids[0]) & set(fold_ids[1]):
        raise ValueError("Outer-fold test questions overlap")
    rows = []
    for (method, layer, params), folds in sorted(grouped.items()):
        if set(folds) != {0, 1}:
            raise ValueError(f"Missing fold: {method}, layer {layer}, {params}")
        row = {"method": method, "layer": layer, "selection": SELECTION,
               "strength": "", "slerp_t": "", "weight_kappa": "", "margin": "",
               **dict(params)}
        for metric in METRICS:
            for fold in (0, 1):
                row[f"{metric}_fold{fold}"] = folds[fold]["metrics"][metric]
            row[metric] = (row[f"{metric}_fold0"] + row[f"{metric}_fold1"]) / 2
        rows.append(row)
    best = {}
    # Rows are sorted by numeric parameter tuples: ties retain the first tuple.
    for row in rows:
        key = (row["method"], row["layer"])
        if key not in best or row["MC1"] > best[key]["MC1"]:
            best[key] = row
    return rows, list(best.values())


def write_summaries(output, reports, config):
    rows, best = aggregate(reports)
    directory = Path(output) / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("all_grid_fold_mean", rows), ("best_per_layer", best)):
        path = directory / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        os.replace(temporary, path)
        write_json(directory / f"{name}.json", {
            "selection": SELECTION,
            "definition": "Arithmetic fold0/fold1 mean at the same parameter tuple; maximize MC1 per methods/layer on test data.",
            "tie_break": "First numeric parameter tuple in sorted order.",
            "requested_config": config, "completed_layers": sorted({r["layer"] for r in rows}),
            "rows": data,
        })


def cells(config, layer):
    """Stable task paths; only the currently requested grid is collected."""
    from itertools import product
    for method in config["methods"]:
        names = ("strength", "slerp_t") if method == "S2S-L" else ("weight_kappa", "margin")
        for values in product(*(config["grid"][name] for name in names)):
            parameters = dict(zip(names, values))
            for fold in (0, 1):
                relative = Path("raw") / method / f"layer{layer}" / f"fold{fold}" / (fingerprint(parameters)[:24] + ".json")
                yield method, parameters, fold, relative
