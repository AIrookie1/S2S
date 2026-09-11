"""Pair-level vector loading for the Fig. 1 MC1 motivation experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairVectorData:
    """Training-pair vectors and answer metadata from one feature fold."""

    vectors: dict[tuple[int, int], np.ndarray]
    shared_mean: np.ndarray
    train_questions: np.ndarray
    choices: np.ndarray
    labels: np.ndarray
    q_indices: np.ndarray
    fold_idx: int | None
    num_train_pairs: int
    mean_pair_norm: float
    mean_vector_norm: float
    max_reconstruction_error: float


def load_pair_vector_data(feature_path: str) -> PairVectorData:
    """Load raw v_ij = h_i+ - h_j- vectors without averaging or normalization."""
    with np.load(feature_path, allow_pickle=False) as features:
        required = {
            "train_q_indices",
            "vector_q_indices",
            "positive_sample_indices",
            "negative_sample_indices",
            "choices",
            "labels",
            "q_indices",
        }
        missing = required.difference(features.files)
        if missing:
            raise KeyError(f"Feature file is missing fields: {sorted(missing)}")

        if "v_T" in features.files:
            all_vectors = np.asarray(features["v_T"], dtype=np.float32)
        elif {"z_pos_raw", "z_neg_raw"}.issubset(features.files):
            all_vectors = np.asarray(features["z_pos_raw"], dtype=np.float32) - np.asarray(
                features["z_neg_raw"], dtype=np.float32
            )
        else:
            raise KeyError("Feature file needs v_T or z_pos_raw/z_neg_raw")

        vector_questions = np.asarray(features["vector_q_indices"]).reshape(-1)
        positive_rows = np.asarray(features["positive_sample_indices"]).reshape(-1)
        negative_rows = np.asarray(features["negative_sample_indices"]).reshape(-1)
        train_questions = np.asarray(features["train_q_indices"], dtype=np.int64).reshape(-1)
        if not (
            len(all_vectors)
            == len(vector_questions)
            == len(positive_rows)
            == len(negative_rows)
        ):
            raise ValueError("Pair vectors and pair metadata must align")

        train_mask = np.isin(vector_questions, train_questions)
        train_vectors = all_vectors[train_mask]
        train_positive_rows = positive_rows[train_mask]
        train_negative_rows = negative_rows[train_mask]
        if len(train_vectors) == 0:
            raise ValueError("No pair vectors belong to the training questions")

        vectors: dict[tuple[int, int], np.ndarray] = {}
        for positive_row, negative_row, vector in zip(
            train_positive_rows, train_negative_rows, train_vectors
        ):
            key = (int(positive_row), int(negative_row))
            if key in vectors:
                raise ValueError(f"Duplicate answer-pair key: {key}")
            vectors[key] = vector

        if {"z_pos_raw", "z_neg_raw"}.issubset(features.files):
            raw_pos = np.asarray(features["z_pos_raw"], dtype=np.float32)[train_mask]
            raw_neg = np.asarray(features["z_neg_raw"], dtype=np.float32)[train_mask]
            reconstruction_error = np.linalg.norm(
                raw_neg + train_vectors - raw_pos, axis=1
            )
            max_reconstruction_error = float(reconstruction_error.max())
        else:
            max_reconstruction_error = float("nan")

        shared_mean = train_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
        fold_idx = (
            int(features["fold_idx"].item()) if "fold_idx" in features.files else None
        )
        return PairVectorData(
            vectors=vectors,
            shared_mean=shared_mean,
            train_questions=np.sort(np.unique(train_questions)),
            choices=np.asarray(features["choices"]).astype(str),
            labels=np.asarray(features["labels"]).reshape(-1),
            q_indices=np.asarray(features["q_indices"]).reshape(-1),
            fold_idx=fold_idx,
            num_train_pairs=len(train_vectors),
            mean_pair_norm=float(np.linalg.norm(train_vectors, axis=1).mean()),
            mean_vector_norm=float(np.linalg.norm(shared_mean)),
            max_reconstruction_error=max_reconstruction_error,
        )


def find_answer_rows(
    data: PairVectorData,
    question_id: int,
    correct_answer: str,
    false_answers: list[str] | None = None,
) -> tuple[int, list[int]]:
    """Map the canonical MC1 correct/false choices to feature sample rows."""
    rows = np.flatnonzero(data.q_indices == int(question_id))
    best_rows = [
        int(row)
        for row in rows
        if data.labels[row] == 1 and data.choices[row] == correct_answer
    ]
    if len(best_rows) != 1:
        raise ValueError(
            f"Question {question_id}: expected one truthful row matching the MC1 "
            f"correct answer {correct_answer!r}, "
            f"found {len(best_rows)}"
        )

    if false_answers is None:
        false_rows = [int(row) for row in rows if data.labels[row] == 0]
    else:
        false_rows = []
        for answer in false_answers:
            matches = [
                int(row)
                for row in rows
                if data.labels[row] == 0 and data.choices[row] == answer
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Question {question_id}: expected one false row matching the "
                    f"MC1 answer {answer!r}, found {len(matches)}"
                )
            false_rows.append(matches[0])
    if not false_rows:
        raise ValueError(f"Question {question_id} has no false MC1 candidates")
    for false_row in false_rows:
        if (best_rows[0], false_row) not in data.vectors:
            raise KeyError(
                f"Missing pair vector for answer rows {(best_rows[0], false_row)}"
            )
    return best_rows[0], false_rows
