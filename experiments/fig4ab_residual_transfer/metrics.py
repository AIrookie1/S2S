"""Pure-numpy utilities for the residual analysis experiment."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def row_normalize(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize rows and return ``(normalized, valid_norm_mask)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape [N, D]")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    norms = np.linalg.norm(x, axis=1)
    valid = norms > eps
    out = np.zeros_like(x, dtype=np.float64)
    out[valid] = x[valid] / norms[valid, None]
    return out, valid


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine/dot product for already-normalized row vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x and y must have matching [N, D] shapes")
    values = np.sum(x * y, axis=1)
    return np.clip(values, -1.0, 1.0)


def angle_degrees_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angular distance in degrees for already-normalized row vectors."""
    return np.degrees(np.arccos(cosine_rows(x, y)))


def constrained_shuffle_indices(
    q_indices: np.ndarray,
    *,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
    max_attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle rows while forbidding donors from the same question.

    If ``groups`` is provided, donors must also come from the same group (used
    for TruthfulQA within-category shuffles).  The function first tries to use
    an exact permutation, so the empirical residual distribution is preserved.
    If an exact constrained permutation is impossible for a group, it falls
    back to sampling with replacement from eligible donors in that group.

    Returns
    -------
    donor_indices:
        Row index of the shuffled donor, or -1 if a group contains only one
        question and therefore has no eligible donor.
    exact_mask:
        Boolean row mask. True means the row came from an exact permutation;
        False means fallback sampling or no donor was necessary.
    """
    q = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    if len(q) == 0:
        raise ValueError("q_indices is empty")
    if groups is None:
        group_values = np.zeros(len(q), dtype=np.int64)
    else:
        group_values = np.asarray(groups)
        if group_values.shape != q.shape:
            raise ValueError("groups must align with q_indices")

    donors = np.full(len(q), -1, dtype=np.int64)
    exact_mask = np.zeros(len(q), dtype=bool)

    for group in np.unique(group_values):
        idx = np.flatnonzero(group_values == group)
        q_group = q[idx]
        if len(np.unique(q_group)) < 2:
            continue

        found = False
        for _ in range(max_attempts):
            permuted = rng.permutation(idx)
            if np.all(q[permuted] != q[idx]):
                donors[idx] = permuted
                exact_mask[idx] = True
                found = True
                break

        if found:
            continue

        # Rare fallback for heavily imbalanced categories.  We still enforce
        # the scientifically important constraint: never take a residual from
        # the same question.
        for row in idx:
            eligible = idx[q[idx] != q[row]]
            if len(eligible):
                donors[row] = int(rng.choice(eligible))

    return donors, exact_mask


def aggregate_mean_by_question(
    q_indices: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average finite row values within question, giving every question one row."""
    q = np.asarray(q_indices, dtype=np.int64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(q) != len(v):
        raise ValueError("q_indices and values must have equal length")
    out_q: list[int] = []
    out_v: list[float] = []
    for question in np.unique(q):
        selected = v[q == question]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            out_q.append(int(question))
            out_v.append(float(np.mean(selected)))
    return np.asarray(out_q, dtype=np.int64), np.asarray(out_v, dtype=np.float64)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over independent question-level rows."""
    x = np.asarray(list(values), dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("bootstrap input contains no finite values")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0,1)")

    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    boot = x[sample_idx].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [tail, 1.0 - tail])
    return mean, float(lo), float(hi)


def equal_count_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Assign deterministic equal-count bins using stable rank order.

    This is robust to many tied cosine values and always returns exactly
    ``min(n_bins, N)`` occupied bins.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        raise ValueError("values is empty")
    if not np.isfinite(x).all():
        raise ValueError("values contains non-finite entries")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    n_bins = min(int(n_bins), len(x))
    order = np.argsort(x, kind="mergesort")
    bins = np.empty(len(x), dtype=np.int64)
    for bin_idx, rows in enumerate(np.array_split(order, n_bins), start=1):
        bins[rows] = bin_idx
    return bins


def one_sided_shuffle_pvalue(observed: float, null_values: Iterable[float]) -> float:
    """Randomization p-value for the alternative ``observed > null``."""
    null = np.asarray(list(null_values), dtype=np.float64).reshape(-1)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        raise ValueError("null_values contains no finite values")
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))
