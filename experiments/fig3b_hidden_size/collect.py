#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from collect_common import checkpoint_scalar, update_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-hidden", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--num-folds", type=int, default=2)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mlp_hidden <= 0:
        raise ValueError("mlp_hidden must be positive")

    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    metrics = evaluation.get("metrics", {})
    try:
        mc = {name: float(metrics[name]) for name in ("MC1", "MC2", "MC3")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{args.eval_json} is missing numeric MC1/MC2/MC3") from exc
    if not all(np.isfinite(value) for value in mc.values()):
        raise ValueError(f"non-finite MC metric in {args.eval_json}: {mc}")

    parameters = evaluation.get("parameters", {})
    eval_kappa = float(parameters.get("weight_kappa", float("nan")))
    eval_margin = float(parameters.get("margin", float("nan")))

    with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
        fold = int(checkpoint_scalar(checkpoint, "fold_idx"))
        hidden = int(checkpoint_scalar(checkpoint, "mlp_hidden"))
        val_fraction = float(checkpoint_scalar(checkpoint, "val_fraction"))
        epochs = int(checkpoint_scalar(checkpoint, "epochs"))
        batch_size = int(checkpoint_scalar(checkpoint, "batch_size"))
        learning_rate = float(checkpoint_scalar(checkpoint, "learning_rate"))
        seed = int(checkpoint_scalar(checkpoint, "seed"))
        positive_mode = str(checkpoint_scalar(checkpoint, "positive_mode"))

    if fold != args.fold:
        raise ValueError(f"checkpoint fold={fold}, requested fold={args.fold}")
    if hidden != args.mlp_hidden:
        raise ValueError(
            f"checkpoint mlp_hidden={hidden}, requested mlp_hidden={args.mlp_hidden}"
        )

    metadata = {
        "experiment": "fig3b_hidden_size",
        "method": "S2S_R",
        "model": args.model,
        "model_label": args.model_label,
        "layer": args.layer,
        "fold_idx": args.fold,
        "num_folds": args.num_folds,
        "x_axis": "mlp_hidden",
        "metrics": ["MC1", "MC2", "MC3"],
        "aggregation": "arithmetic mean across fold0/fold1",
    }
    record = {
        "mlp_hidden": args.mlp_hidden,
        **mc,
        "val_fraction": val_fraction,
        "positive_mode": positive_mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "eval_weight_kappa": eval_kappa,
        "eval_margin": eval_margin,
        "checkpoint": str(args.checkpoint),
        "evaluation": str(args.eval_json),
    }
    update_summary(
        args.output,
        metadata,
        record,
        identity_keys=("mlp_hidden",),
        sort_keys=("mlp_hidden",),
    )
    print(
        f"mlp_hidden={args.mlp_hidden} fold={args.fold} "
        f"MC1={mc['MC1']:.6f} MC2={mc['MC2']:.6f} MC3={mc['MC3']:.6f}"
    )


if __name__ == "__main__":
    main()
