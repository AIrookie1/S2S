"""Prepare raw class-centroid additive vectors using each outer training half only."""
import argparse
from pathlib import Path


def prepare(features, output_dir):
    import numpy as np
    with np.load(features, allow_pickle=False) as data:
        h = np.asarray(data["activations"], dtype=np.float32)
        labels = data["labels"].reshape(-1)
        qids = data["q_indices"].reshape(-1)
    if h.ndim != 2 or len(h) != len(labels) or len(h) != len(qids):
        raise ValueError("Feature arrays do not align")
    halves = np.array_split(np.unique(qids), 2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in (0, 1):
        train, test = halves[fold], halves[1 - fold]
        mask = np.isin(qids, train)
        positive, negative = h[mask & (labels == 1)], h[mask & (labels == 0)]
        if not len(positive) or not len(negative):
            raise ValueError("Each training half must contain both answer classes")
        vector = positive.mean(axis=0) - negative.mean(axis=0)
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite mean difference")
        np.savez_compressed(output_dir / f"fold{fold}.npz", v=vector, fold_idx=fold,
                            train_q_indices=train, test_q_indices=test,
                            definition="raw unweighted answer-class centroid difference on outer training questions")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    prepare(a.features, a.output_dir)


if __name__ == "__main__":
    main()
