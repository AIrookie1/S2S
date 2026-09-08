#!/usr/bin/env python3
"""Add one SVD-energy evaluation result to the Fig. 3 summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", type=float, required=True)
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

    if args.output.exists():
        summary = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        summary = {
            "experiment": "fig3_svd_energy_mc1",
            "method": "Similarity-Graph Adaptive 3D Spherical Steering",
            "x_axis": "SVD_ENERGY",
            "y_axis": "MC1",
            "results": [],
        }

    record = {
        "svd_energy": args.energy,
        "rank": rank,
        "MC1": mc1,
        "checkpoint": str(args.checkpoint),
        "evaluation": str(args.eval_json),
    }
    records = [
        item
        for item in summary.get("results", [])
        if float(item["svd_energy"]) != args.energy
    ]
    records.append(record)
    summary["results"] = sorted(records, key=lambda item: float(item["svd_energy"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"SVD_ENERGY={args.energy:.1f} | rank={rank} | MC1={mc1:.6f}")


if __name__ == "__main__":
    main()
