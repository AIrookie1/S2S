"""Resumable joint kappa/margin search for the mean-only control."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path


def validate_grid(kappas, margins):
    if not kappas or not margins:
        raise ValueError("provide both --vavg_kappas and --vavg_margins")
    if any(not math.isfinite(k) or k <= 0 for k in kappas):
        raise ValueError("all kappa values must be finite and positive")
    if any(not math.isfinite(m) for m in margins):
        raise ValueError("all margins must be finite")
    return list(dict.fromkeys(kappas)), list(dict.fromkeys(margins))


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _valid_row(row, kappa, margin):
    if not isinstance(row, dict) or row.get("variant") != "vavg_aw_gate":
        return False
    if row.get("weight_kappa") != kappa or row.get("margin") != margin:
        return False
    if any(type(row.get(key)) not in (float, int) or not math.isfinite(row[key])
           or not 0 <= row[key] <= 1 for key in ("MC1", "MC2", "MC3", "Avg", "steer_token_ratio")):
        return False
    return math.isclose(row["Avg"], sum(row[k] for k in ("MC1", "MC2", "MC3")) / 3,
                        rel_tol=0, abs_tol=1e-10)


def _write_summary(output_dir, context, rows, total, metric):
    # Equal primary scores use Avg, then MC2/MC3, then smaller kappa/margin.
    ranked = sorted(rows, key=lambda r: (-r[metric], -r["Avg"], -r["MC2"],
                                        -r["MC3"], r["weight_kappa"], r["margin"]))
    metadata = {"context": context, "selection_metric": metric,
                "selection_split": context["eval_split"],
                "tie_break": "Avg, MC2, MC3 descending; kappa, margin ascending",
                "completed": len(rows), "total": total, "complete": len(rows) == total,
                "selection_scope": "best on the evaluated split within this grid"}
    _write_json(output_dir / "grid_summary.json", {"metadata": metadata, "rows": ranked})
    _write_json(output_dir / "best_config.json", {"metadata": metadata, "best": ranked[0]})
    fields = ["weight_kappa", "margin", "MC1", "MC2", "MC3", "Avg", "steer_token_ratio"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(ranked)
    _write_text(output_dir / "grid_summary.csv", stream.getvalue())
    return ranked[0]


def run_grid(*, kappas, margins, output_dir, context, evaluate,
             selection_metric="MC1", resume=False, save_details=False):
    """Evaluate each cell once; checkpoint and re-rank after every result.

    ``evaluate(kappa, margin)`` returns the component evaluator's row/detail.
    Model loading happens in the caller, once for the entire search.
    """
    kappas, margins = validate_grid(kappas, margins)
    if selection_metric not in {"MC1", "MC2", "MC3", "Avg"}:
        raise ValueError("unsupported selection metric")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"context": context, "kappas": kappas, "margins": margins}
    config_path = output_dir / "grid_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory contains a different grid/context; choose another output_dir")
    _write_json(config_path, config)
    rows = []
    total = len(kappas) * len(margins)
    for kappa in kappas:
        for margin in margins:
            tag = f"kappa{kappa}_margin{margin}"
            cell_path = output_dir / "cells" / f"{tag}.json"
            row = None
            if resume:
                try:
                    cell = json.loads(cell_path.read_text())
                    candidate = cell.get("row")
                    if cell.get("context") == context and _valid_row(candidate, kappa, margin):
                        if not save_details or (output_dir / "details" / f"{tag}.json").exists():
                            row = candidate
                except (OSError, ValueError, AttributeError):
                    pass
            if row is None:
                print(f"[grid {len(rows)+1}/{total}] kappa={kappa} margin={margin}", flush=True)
                row, detail = evaluate(kappa, margin)
                row = {**row, "weight_kappa": kappa, "margin": margin}
                if not _valid_row(row, kappa, margin):
                    raise ValueError(f"invalid grid result: {row}")
                if save_details:
                    _write_json(output_dir / "details" / f"{tag}.json", detail)
                _write_json(cell_path, {"context": context, "row": row})
            else:
                print(f"[reuse] kappa={kappa} margin={margin}", flush=True)
            rows.append(row)
            best = _write_summary(output_dir, context, rows, total, selection_metric)
            print(f"[best so far] {selection_metric}={best[selection_metric]:.6f} "
                  f"kappa={best['weight_kappa']} margin={best['margin']} "
                  f"({len(rows)}/{total})", flush=True)
    return best
