"""Pair-level vector loading for the steering-vector ablation.

The experiment keeps one base ``*_dynSVD_pairs.npz`` file per layer.  Fold
membership is reconstructed in memory from ``q_indices`` using the same
deterministic ``np.array_split`` rule that was previously used to materialize
``*_fold0.npz`` / ``*_fold1.npz`` copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairVectorData:
    """Pair vectors restricted to the training questions for one outer fold."""

    vectors: dict[tuple[int, int], np.ndarray]
    raw_positive_states: dict[tuple[int, int], np.ndarray]
    raw_negative_states: dict[tuple[int, int], np.ndarray]
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


def _derive_train_questions(
    q_indices: np.ndarray,
    *,
    fold_idx: int,
    num_folds: int,
) -> np.ndarray:
    """Reproduce the former on-disk fold helper without writing fold files.

    The selected split ``fold_idx`` is the held-out/test portion.  The returned
    training questions are the union of all other splits.  This is exactly the
    rule previously used by ``prepare_fold_features.py``.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if not 0 <= fold_idx < num_folds:
        raise ValueError(
            f"fold_idx must satisfy 0 <= fold_idx < num_folds; "
            f"got fold_idx={fold_idx}, num_folds={num_folds}"
        )

    questions = np.sort(np.unique(np.asarray(q_indices, dtype=np.int64).reshape(-1)))
    if len(questions) < num_folds:
        raise ValueError(
            f"Need at least {num_folds} questions for {num_folds} folds; "
            f"found {len(questions)}"
        )

    splits = [
        np.asarray(split, dtype=np.int64)
        for split in np.array_split(questions, num_folds)
    ]
    train_parts = [split for i, split in enumerate(splits) if i != fold_idx]
    train_questions = np.sort(np.concatenate(train_parts))
    if len(train_questions) == 0:
        raise ValueError(f"Fold {fold_idx} has no training questions")
    return train_questions


def load_pair_vector_data(
    feature_path: str,
    *,
    fold_idx: int | None = None,
    num_folds: int = 2,
) -> PairVectorData:
    """Load raw ``v_ij = h_i+ - h_j-`` vectors without averaging/normalization.

    Preferred mode
    --------------
    Pass ``fold_idx`` and read the single base ``*_dynSVD_pairs.npz`` file.
    Training-question membership is reconstructed in memory, so no fold-specific
    NPZ files are created or required.

    Backward-compatible mode
    ------------------------
    If ``fold_idx`` is omitted, a legacy file containing ``train_q_indices`` is
    still accepted.  This keeps the original uploaded unit tests and any older
    artifacts readable, but the current experiment runner never writes or needs
    those fold files.
    """
    with np.load(feature_path, allow_pickle=False) as features:
        required = {
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

        q_indices = np.asarray(features["q_indices"]).reshape(-1)
        if fold_idx is None:
            if "train_q_indices" not in features.files:
                raise KeyError(
                    "Base pair feature has no train_q_indices. "
                    "Pass fold_idx so the fold can be reconstructed in memory."
                )
            train_questions = np.asarray(
                features["train_q_indices"], dtype=np.int64
            ).reshape(-1)
            effective_fold_idx = (
                int(features["fold_idx"].item())
                if "fold_idx" in features.files
                else None
            )
        else:
            effective_fold_idx = int(fold_idx)
            train_questions = _derive_train_questions(
                q_indices,
                fold_idx=effective_fold_idx,
                num_folds=int(num_folds),
            )

        if "v_T" in features.files:
            all_vectors = np.asarray(features["v_T"], dtype=np.float32)
        elif {"z_pos_raw", "z_neg_raw"}.issubset(features.files):
            all_vectors = (
                np.asarray(features["z_pos_raw"], dtype=np.float32)
                - np.asarray(features["z_neg_raw"], dtype=np.float32)
            )
        else:
            raise KeyError("Feature file needs v_T or z_pos_raw/z_neg_raw")

        vector_questions = np.asarray(features["vector_q_indices"]).reshape(-1)
        positive_rows = np.asarray(features["positive_sample_indices"]).reshape(-1)
        negative_rows = np.asarray(features["negative_sample_indices"]).reshape(-1)
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
            train_positive_rows,
            train_negative_rows,
            train_vectors,
        ):
            key = (int(positive_row), int(negative_row))
            if key in vectors:
                raise ValueError(f"Duplicate answer-pair key: {key}")
            vectors[key] = vector

        raw_positive_states: dict[tuple[int, int], np.ndarray] = {}
        raw_negative_states: dict[tuple[int, int], np.ndarray] = {}
        if {"z_pos_raw", "z_neg_raw"}.issubset(features.files):
            raw_pos = np.asarray(features["z_pos_raw"], dtype=np.float32)[train_mask]
            raw_neg = np.asarray(features["z_neg_raw"], dtype=np.float32)[train_mask]
            if len(raw_pos) != len(train_vectors) or len(raw_neg) != len(train_vectors):
                raise ValueError("Raw pair endpoints must align with pair vectors")
            for positive_row, negative_row, z_pos, z_neg in zip(
                train_positive_rows,
                train_negative_rows,
                raw_pos,
                raw_neg,
            ):
                key = (int(positive_row), int(negative_row))
                raw_positive_states[key] = z_pos
                raw_negative_states[key] = z_neg
            reconstruction_error = np.linalg.norm(
                raw_neg + train_vectors - raw_pos,
                axis=1,
            )
            max_reconstruction_error = float(reconstruction_error.max())
        else:
            max_reconstruction_error = float("nan")

        shared_mean = train_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
        return PairVectorData(
            vectors=vectors,
            raw_positive_states=raw_positive_states,
            raw_negative_states=raw_negative_states,
            shared_mean=shared_mean,
            train_questions=np.sort(np.unique(train_questions)),
            choices=np.asarray(features["choices"]).astype(str),
            labels=np.asarray(features["labels"]).reshape(-1),
            q_indices=q_indices,
            fold_idx=effective_fold_idx,
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
