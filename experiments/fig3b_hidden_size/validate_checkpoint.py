#!/usr/bin/env python3
"""Validate that a cached S2S_R checkpoint matches one ablation cell."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data.files:
        raise KeyError(key)
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar: shape={value.shape}")
    return value.reshape(()).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-hidden", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--num-folds", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, required=True)
    parser.add_argument("--positive-mode", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    try:
        with np.load(args.checkpoint, allow_pickle=False) as data:
            actual = {
                "mlp_hidden": int(scalar(data, "mlp_hidden")),
                "fold_idx": int(scalar(data, "fold_idx")),
                "num_folds": int(scalar(data, "num_folds")),
                "val_fraction": float(scalar(data, "val_fraction")),
                "positive_mode": str(scalar(data, "positive_mode")),
                "epochs": int(scalar(data, "epochs")),
                "batch_size": int(scalar(data, "batch_size")),
                "learning_rate": float(scalar(data, "learning_rate")),
                "seed": int(scalar(data, "seed")),
            }
    except Exception as exc:
        print(f"invalid checkpoint {args.checkpoint}: {exc}")
        raise SystemExit(1)

    expected = {
        "mlp_hidden": args.mlp_hidden,
        "fold_idx": args.fold,
        "num_folds": args.num_folds,
        "val_fraction": args.val_fraction,
        "positive_mode": args.positive_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }

    mismatches: list[str] = []
    for key, wanted in expected.items():
        got = actual[key]
        if isinstance(wanted, float):
            matches = math.isclose(float(got), wanted, rel_tol=1e-6, abs_tol=1e-8)
        else:
            matches = got == wanted
        if not matches:
            mismatches.append(f"{key}: cached={got!r}, requested={wanted!r}")

    if mismatches:
        print(f"stale checkpoint {args.checkpoint}: " + "; ".join(mismatches))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
