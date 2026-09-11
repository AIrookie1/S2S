"""Create fold-aware pair feature files for the steering-vector ablation.

The base Dyn3D feature extractor stores all answer-pair vectors in one NPZ.
The uploaded vector_utils.py intentionally consumes fold-aware files containing
``train_q_indices``.  This helper adds a deterministic outer-fold split without
altering any original feature arrays.

For compatibility with train_dynSVD.py's fold-artifact convention,
``test_q_indices`` is the selected fold and ``train_q_indices`` is its
complement.  With the current two-fold experiments the two training sets are
complementary and cover the full question set exactly once across fold files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add deterministic fold metadata to a Dyn3D pair-feature NPZ")
    p.add_argument("--input", required=True, help="Base *_dynSVD_pairs.npz feature file")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.num_folds < 2:
        p.error("--num_folds must be >= 2")
    if a.folds is not None and any(f < 0 or f >= a.num_folds for f in a.folds):
        p.error("each --folds value must satisfy 0 <= fold < num_folds")
    return a


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(src, allow_pickle=False) as data:
        required = {
            "q_indices", "vector_q_indices", "positive_sample_indices",
            "negative_sample_indices", "choices", "labels",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Base feature file is missing fields: {sorted(missing)}")
        payload = {key: np.asarray(data[key]) for key in data.files}

    questions = np.sort(np.unique(np.asarray(payload["q_indices"], dtype=np.int64).reshape(-1)))
    if len(questions) < args.num_folds:
        raise ValueError(f"Need at least {args.num_folds} questions, found {len(questions)}")
    split = [np.asarray(x, dtype=np.int64) for x in np.array_split(questions, args.num_folds)]
    folds = args.folds if args.folds is not None else list(range(args.num_folds))

    stem = src.stem
    for fold_idx in folds:
        selected = split[fold_idx]
        train = np.sort(np.concatenate([x for i, x in enumerate(split) if i != fold_idx]))
        out = out_dir / f"{stem}_fold{fold_idx}.npz"
        if out.exists() and not args.overwrite:
            print(f"Exists, keeping: {out}")
            continue
        fold_payload = dict(payload)
        fold_payload.update(
            train_q_indices=train,
            test_q_indices=np.sort(selected),
            fold_idx=np.asarray(fold_idx, dtype=np.int64),
            num_folds=np.asarray(args.num_folds, dtype=np.int64),
            source_feature_path=np.asarray(str(src)),
        )
        np.savez_compressed(out, **fold_payload)
        print(f"Saved {out} | train={len(train)} test={len(selected)}")


if __name__ == "__main__":
    main()
