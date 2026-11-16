from __future__ import annotations

import numpy as np

from experiments.fig2bc_l2s_reliability.metrics import (
    aggregate_by_question,
    mean_vector_from_questions,
    rowwise_vector_metrics,
)


def test_rowwise_metrics_direction_and_requested_relative_norm_error():
    target = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = np.asarray([[2.0, 0.0], [0.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    cosine, magnitude = rowwise_vector_metrics(pred, target)

    np.testing.assert_allclose(cosine, [1.0, -1.0, 0.0], atol=1e-7)
    # 2 vs 1 -> 1.0; 1 vs 2 -> 0.5; 0 vs 5 -> 1.0
    np.testing.assert_allclose(magnitude, [1.0, 0.5, 1.0], atol=1e-7)


def test_question_aggregation_gives_each_question_one_row():
    q = np.asarray([10, 10, 10, 20], dtype=np.int64)
    cos = np.asarray([1.0, 0.0, -1.0, 0.5])
    mag = np.asarray([0.1, 0.2, 0.3, 0.8])
    rows = aggregate_by_question(q, cos, mag)

    assert [row["q_idx"] for row in rows] == [10, 20]
    assert rows[0]["num_pairs"] == 3
    assert np.isclose(rows[0]["cosine_similarity"], 0.0)
    assert np.isclose(rows[0]["magnitude_relative_error"], 0.2)
    assert rows[1]["num_pairs"] == 1
    assert np.isclose(rows[1]["cosine_similarity"], 0.5)


def test_mean_vector_uses_only_l2s_training_questions():
    vectors = np.asarray(
        [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
        dtype=np.float32,
    )
    q = np.asarray([0, 0, 1], dtype=np.int64)
    mean = mean_vector_from_questions(vectors, q, [0])
    np.testing.assert_allclose(mean, [2.0, 0.0], atol=1e-7)
