#!/usr/bin/env python3
"""Collect MC1/MC2/MC3 for one Table 1 loss-ablation setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vector-loss-weight", type=float, required=True)
    parser.add_argument("--scale-loss-weight", type=float, required=True)
    parser.add_argument("--cosine-loss-weight", type=float, required=True)
    parser.add_argument("--val-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    try:
        metrics = {
            name: float(evaluation["metrics"][name])
            for name in ("MC1", "MC2", "MC3")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{args.eval_json} must contain numeric metrics MC1, MC2 and MC3"
        ) from exc

    with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
        rank = int(np.asarray(checkpoint["rank"]).item())
        actual_fold = int(np.asarray(checkpoint["fold_idx"]).item())
    if actual_fold != args.fold:
        raise ValueError(
            f"Requested fold={args.fold}, checkpoint stores fold={actual_fold}"
        )

    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("experiment") != "table1_loss_ablation":
            raise ValueError(f"Unexpected experiment metadata in {args.output}")
    else:
        payload = {
            "experiment": "table1_loss_ablation",
            "method": "Similarity-Graph Adaptive 3D Spherical Steering",
            "description": (
                "Ablation of vector MSE, scale MSE and cosine-direction losses; "
                "each record reports MC1, MC2 and MC3."
            ),
            "x_axis": "loss_configuration",
            "metrics": ["MC1", "MC2", "MC3"],
            "results": [],
        }

    record = {
        "model": args.model,
        "model_label": args.model_label,
        "layer": args.layer,
        "fold_idx": args.fold,
        "num_folds": args.num_folds,
        "loss_configuration": args.config,
        "loss_weights": {
            "mse": args.vector_loss_weight,
            "scale": args.scale_loss_weight,
            "cosine": args.cosine_loss_weight,
        },
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "rank": rank,
        **metrics,
        "checkpoint": str(args.checkpoint),
        "evaluation": str(args.eval_json),
    }
    identity = (
        args.model,
        args.fold,
        args.config,
        args.val_fraction,
        args.seed,
    )
    records = []
    for item in payload.get("results", []):
        old_identity = (
            item.get("model"),
            int(item.get("fold_idx", -1)),
            item.get("loss_configuration"),
            float(item.get("val_fraction")),
            int(item.get("seed", 0)),
        )
        if old_identity != identity:
            records.append(item)
    records.append(record)
    payload["results"] = sorted(
        records,
        key=lambda item: (
            str(item["model"]),
            int(item["fold_idx"]),
            ["mse_only", "mse_scale", "cos_only", "cos_scale", "all"].index(
                str(item["loss_configuration"])
            )
            if str(item["loss_configuration"])
            in {"mse_only", "mse_scale", "cos_only", "cos_scale", "all"}
            else str(item["loss_configuration"]),
        ),
    )
    payload["num_results"] = len(payload["results"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(
        f"model={args.model} | fold={args.fold} | {args.config} | "
        f"MC1={metrics['MC1']:.6f} MC2={metrics['MC2']:.6f} MC3={metrics['MC3']:.6f}"
    )


if __name__ == "__main__":
    main()
