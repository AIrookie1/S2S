from __future__ import annotations

import numpy as np

from experiments.residual_analysis.metrics import (
    aggregate_mean_by_question,
    bootstrap_mean_ci,
    constrained_shuffle_indices,
    equal_count_bins,
    one_sided_shuffle_pvalue,
    row_normalize,
)


def test_row_normalize_marks_near_zero_rows():
    x = np.asarray([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]])
    normalized, valid = row_normalize(x)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[2], [0.0, -1.0])
    assert valid.tolist() == [True, False, True]


def test_constrained_global_shuffle_never_uses_same_question():
    q = np.repeat(np.arange(12), 3)
    donors, _ = constrained_shuffle_indices(q, rng=np.random.default_rng(7))
    assert np.all(donors >= 0)
    assert np.all(q[donors] != q)
    assert sorted(donors.tolist()) == list(range(len(q)))


def test_constrained_category_shuffle_stays_in_category_and_changes_question():
    q = np.repeat(np.arange(8), 2)
    category = np.repeat(np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"]), 2)
    donors, _ = constrained_shuffle_indices(
        q,
        groups=category,
        rng=np.random.default_rng(11),
    )
    assert np.all(donors >= 0)
    assert np.all(q[donors] != q)
    assert np.all(category[donors] == category)


def test_single_question_group_is_explicitly_unavailable():
    q = np.asarray([1, 1, 2, 3])
    group = np.asarray(["singleton", "singleton", "other", "other"])
    donors, exact = constrained_shuffle_indices(
        q,
        groups=group,
        rng=np.random.default_rng(0),
    )
    assert donors[:2].tolist() == [-1, -1]
    assert exact[:2].tolist() == [False, False]
    assert q[donors[2]] != q[2]
    assert q[donors[3]] != q[3]


def test_question_aggregation_and_equal_count_bins():
    q, values = aggregate_mean_by_question(
        np.asarray([0, 0, 1, 1, 1]),
        np.asarray([1.0, 3.0, 2.0, 4.0, 6.0]),
    )
    assert q.tolist() == [0, 1]
    np.testing.assert_allclose(values, [2.0, 4.0])

    bins = equal_count_bins(np.asarray([0.5, -1.0, 0.5, 2.0, 1.0]), n_bins=5)
    assert sorted(bins.tolist()) == [1, 2, 3, 4, 5]


def test_bootstrap_and_shuffle_pvalue_are_well_formed():
    mean, lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_bootstrap=200, seed=3)
    assert np.isclose(mean, 2.5)
    assert lo <= mean <= hi
    p = one_sided_shuffle_pvalue(0.8, [0.1, 0.2, 0.7, 0.9])
    assert np.isclose(p, 2.0 / 5.0)
