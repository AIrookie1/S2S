from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.steering_vector_ablation.vector_utils import (
    find_answer_rows,
    load_pair_vector_data,
)


class PairVectorDataTest(unittest.TestCase):
    def test_raw_pair_vectors_shared_mean_and_answer_mapping(self) -> None:
        choices = np.asarray(
            ["best zero", "false zero a", "false zero b", "best one", "false one"]
        )
        labels = np.asarray([1, 0, 0, 1, 0], dtype=np.int64)
        q_indices = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)

        positive_rows = np.asarray([0, 0, 3], dtype=np.int64)
        negative_rows = np.asarray([1, 2, 4], dtype=np.int64)
        vector_questions = np.asarray([0, 0, 1], dtype=np.int64)
        z_neg_raw = np.asarray([[1.0, 2.0], [2.0, 4.0], [8.0, 8.0]], dtype=np.float32)
        v_t = np.asarray([[2.0, 0.0], [0.0, 4.0], [9.0, 9.0]], dtype=np.float32)
        z_pos_raw = z_neg_raw + v_t

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_fold.npz"
            np.savez(
                path,
                train_q_indices=np.asarray([0], dtype=np.int64),
                vector_q_indices=vector_questions,
                positive_sample_indices=positive_rows,
                negative_sample_indices=negative_rows,
                choices=choices,
                labels=labels,
                q_indices=q_indices,
                v_T=v_t,
                z_pos_raw=z_pos_raw,
                z_neg_raw=z_neg_raw,
                fold_idx=np.asarray(0, dtype=np.int64),
            )

            data = load_pair_vector_data(str(path))

        self.assertEqual(set(data.vectors), {(0, 1), (0, 2)})
        np.testing.assert_array_equal(data.vectors[(0, 1)], [2.0, 0.0])
        np.testing.assert_array_equal(data.vectors[(0, 2)], [0.0, 4.0])
        np.testing.assert_array_equal(data.shared_mean, [1.0, 2.0])
        self.assertEqual(data.num_train_pairs, 2)
        self.assertEqual(data.fold_idx, 0)
        self.assertAlmostEqual(data.max_reconstruction_error, 0.0)
        self.assertEqual(
            find_answer_rows(
                data, 0, "best zero", ["false zero b", "false zero a"]
            ),
            (0, [2, 1]),
        )

    def test_missing_best_to_false_pair_is_rejected(self) -> None:
        choices = np.asarray(["best", "false a", "false b"])
        labels = np.asarray([1, 0, 0], dtype=np.int64)
        q_indices = np.asarray([0, 0, 0], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.npz"
            np.savez(
                path,
                train_q_indices=np.asarray([0], dtype=np.int64),
                vector_q_indices=np.asarray([0], dtype=np.int64),
                positive_sample_indices=np.asarray([0], dtype=np.int64),
                negative_sample_indices=np.asarray([1], dtype=np.int64),
                choices=choices,
                labels=labels,
                q_indices=q_indices,
                v_T=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            data = load_pair_vector_data(str(path))

        with self.assertRaises(KeyError):
            find_answer_rows(data, 0, "best", ["false a", "false b"])


if __name__ == "__main__":
    unittest.main()
