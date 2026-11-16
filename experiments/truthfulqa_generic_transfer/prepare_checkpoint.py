"""Prepare a frozen TruthfulQA MLP and its original fold-specific mean for transfer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def validate_source(path: Path, model: str, layer: int) -> None:
    with np.load(path, allow_pickle=False) as data:
        if str(data["model_name"].item()) != model or int(data["layer"].item()) != layer:
            raise ValueError("TruthfulQA features do not match requested model/layer")
        if str(data["dataset_name"].item()) != "truthful_qa":
            raise ValueError("MLP source features must come from TruthfulQA")
        if {"train_q_indices", "test_q_indices"}.issubset(data.files):
            raise ValueError("Use unsplit TruthfulQA features so fold 0 and fold 1 remain distinct")


def prepare(args) -> None:
    validate_source(args.truthfulqa_feature, args.model, args.layer)
    with np.load(args.source_checkpoint, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    if int(payload["fold_idx"].item()) != args.source_fold or int(payload["num_folds"].item()) != 2:
        raise ValueError("Source checkpoint must match the requested TruthfulQA fold out of two")
    if Path(str(payload["feature_path"].item())).resolve() != args.truthfulqa_feature.resolve():
        raise ValueError("Source checkpoint was trained from a different feature file")
    if str(payload["positive_mode"].item()) != "mean":
        raise ValueError("This experiment requires positive_mode=mean")
    with np.load(args.truthfulqa_feature, allow_pickle=False) as data:
        expected = np.array_split(np.unique(data["q_indices"]), 2)[args.source_fold]
    if not np.array_equal(payload["outer_train_q_indices"], expected):
        raise ValueError("Source checkpoint question split does not match its TruthfulQA fold")
    dim = int(payload["hidden_dim"].item())
    hidden = int(payload["mlp_hidden"].item())
    for key, shape in {
        "mlp_fc1.weight": (hidden, dim), "mlp_fc1.bias": (hidden,),
        "mlp_fc2.weight": (dim, hidden), "mlp_fc2.bias": (dim,),
        "v_avg": (dim,),
    }.items():
        if payload[key].shape != shape or not np.isfinite(payload[key]).all():
            raise ValueError(f"Invalid source checkpoint array: {key}")
    mean_q = np.asarray(payload["mean_train_q_indices"]).reshape(-1)
    mlp_q = np.asarray(payload["mlp_train_q_indices"]).reshape(-1)
    if (not len(mean_q) or not len(mlp_q)
            or np.intersect1d(mean_q, mlp_q).size
            or not np.array_equal(np.union1d(mean_q, mlp_q), expected)):
        raise ValueError("TruthfulQA mean/MLP subsets must partition the selected source fold")
    if np.linalg.norm(payload["v_avg"]) == 0:
        raise ValueError("TruthfulQA v_avg must be nonzero")
    payload.update({
        "experiment": np.asarray("truthfulqa_generic_transfer"),
        "model_name": np.asarray(args.model), "layer": np.asarray(args.layer),
        "mlp_source_dataset": np.asarray("truthful_qa"),
        "mlp_source_checkpoint": np.asarray(str(args.source_checkpoint.resolve())),
        "v_avg_source_dataset": np.asarray("truthful_qa"),
        "v_avg_feature_path": np.asarray(str(args.truthfulqa_feature.resolve())),
        "target_dataset": np.asarray(args.dataset),
        "split_metadata_scope": np.asarray("Training/split/metrics fields describe the source TruthfulQA training run"),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved {args.output}: TruthfulQA fold {args.source_fold} MLP + TruthfulQA mean ({len(mean_q)} questions), target={args.dataset}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate-source", action="store_true")
    p.add_argument("--truthfulqa-feature", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--dataset")
    p.add_argument("--source-fold", type=int, choices=[0, 1])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.validate_source:
        validate_source(args.truthfulqa_feature, args.model, args.layer)
    else:
        for key in ("source_checkpoint", "dataset", "source_fold", "output"):
            if getattr(args, key) is None:
                p.error(f"--{key.replace('_', '-')} is required when preparing a checkpoint")
        prepare(args)


if __name__ == "__main__":
    main()
