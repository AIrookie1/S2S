"""Vector-reliability metrics for the L2S reliability experiment."""

from __future__ import annotations

from typing import Iterable
import numpy as np


def rowwise_vector_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine similarity and relative norm error for every pair.

    Relative norm error is exactly

        abs(||v_pred|| - ||v_T||) / ||v_T||

    as specified for the main reliability figure.  Zero-norm predictions are
    allowed (cosine=0, relative norm error=1); zero-norm ground-truth targets
    are rejected because both metrics would be undefined.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or target.shape != predicted.shape:
        raise ValueError("predicted and target must have matching [N, D] shapes")
    if len(predicted) == 0:
        raise ValueError("metric input is empty")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("metric input contains non-finite values")

    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    if np.any(target_norm <= eps):
        count = int(np.sum(target_norm <= eps))
        raise ValueError(f"{count} ground-truth steering vectors have near-zero norm")

    dot = np.sum(predicted * target, axis=1)
    cosine = dot / (np.maximum(pred_norm, eps) * target_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    magnitude_error = np.abs(pred_norm - target_norm) / target_norm
    return cosine.astype(np.float64), magnitude_error.astype(np.float64)


def aggregate_by_question(
    q_indices: np.ndarray,
    cosine: np.ndarray,
    magnitude_error: np.ndarray,
) -> list[dict[str, float | int]]:
    """Average pair-level metrics within each question.

    This avoids treating multiple positive×negative answer pairs from the same
    TruthfulQA question as statistically independent samples in the figure.
    """
    q_indices = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    magnitude_error = np.asarray(magnitude_error, dtype=np.float64).reshape(-1)
    if not (len(q_indices) == len(cosine) == len(magnitude_error)):
        raise ValueError("question IDs and metric arrays must have equal length")
    if len(q_indices) == 0:
        raise ValueError("question aggregation input is empty")

    rows: list[dict[str, float | int]] = []
    for q in np.unique(q_indices):
        mask = q_indices == q
        rows.append(
            {
                "q_idx": int(q),
                "num_pairs": int(mask.sum()),
                "cosine_similarity": float(np.mean(cosine[mask])),
                "magnitude_relative_error": float(np.mean(magnitude_error[mask])),
            }
        )
    return rows


def mean_vector_from_questions(
    vectors: np.ndarray,
    q_indices: np.ndarray,
    train_questions: Iterable[int],
) -> np.ndarray:
    """Compute the raw pair-vector mean using only L2S optimization questions."""
    vectors = np.asarray(vectors, dtype=np.float32)
    q_indices = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    train_questions = np.asarray(list(train_questions), dtype=np.int64).reshape(-1)
    if vectors.ndim != 2 or len(vectors) != len(q_indices):
        raise ValueError("vectors and q_indices are misaligned")
    if len(train_questions) == 0:
        raise ValueError("L2S training-question set is empty")
    mask = np.isin(q_indices, train_questions)
    missing = np.setdiff1d(train_questions, np.unique(q_indices[mask]))
    if len(missing):
        raise ValueError(f"feature file misses L2S training questions: {missing[:10].tolist()}")
    mean = vectors[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or np.linalg.norm(mean) <= 1e-8:
        raise ValueError("mean steering vector is non-finite or near zero")
    return mean
