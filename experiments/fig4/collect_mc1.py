#!/usr/bin/env python3
"""Collect one validation-fraction result into a model/fold summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--num-folds", type=int, default=2)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    try:
        mc1 = float(evaluation["metrics"]["MC1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric metrics.MC1 in {args.eval_json}") from exc

    with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
        rank = int(np.asarray(checkpoint["rank"]).item())
        actual_fold = int(np.asarray(checkpoint["fold_idx"]).item())

    if actual_fold != args.fold:
        raise ValueError(
            f"Requested fold={args.fold}, checkpoint stores fold={actual_fold}"
        )

    if args.output.exists():
        summary = json.loads(args.output.read_text(encoding="utf-8"))
        if summary.get("model") != args.model or int(summary.get("fold_idx", -1)) != args.fold:
            raise ValueError(f"Summary metadata mismatch in {args.output}")
    else:
        summary = {
            "experiment": "fig4_val_fraction_mc1",
            "method": "Similarity-Graph Adaptive 3D Spherical Steering",
            "model": args.model,
            "model_label": args.model_label,
            "layer": args.layer,
            "fold_idx": args.fold,
            "num_folds": args.num_folds,
            "x_axis": "VAL_FRACTION",
            "y_axis": "MC1",
            "results": [],
        }

    record = {
        "val_fraction": args.val_fraction,
        "rank": rank,
        "MC1": mc1,
        "checkpoint": str(args.checkpoint),
        "evaluation": str(args.eval_json),
    }
    records = [
        item
        for item in summary.get("results", [])
        if float(item["val_fraction"]) != args.val_fraction
    ]
    records.append(record)
    summary["results"] = sorted(
        records, key=lambda item: float(item["val_fraction"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(args.output)

    print(
        f"model={args.model} | fold={args.fold} | "
        f"VAL_FRACTION={args.val_fraction:.1f} | rank={rank} | MC1={mc1:.6f}"
    )


if __name__ == "__main__":
    main()
