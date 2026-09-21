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
        np.testing.assert_array_equal(data.raw_negative_states[(0, 1)], z_neg_raw[0])
        np.testing.assert_array_equal(data.raw_positive_states[(0, 1)], z_pos_raw[0])
        np.testing.assert_array_equal(data.raw_negative_states[(0, 2)], z_neg_raw[1])
        np.testing.assert_array_equal(data.raw_positive_states[(0, 2)], z_pos_raw[1])
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

        self.assertEqual(data.raw_positive_states, {})
        self.assertEqual(data.raw_negative_states, {})
        with self.assertRaises(KeyError):
            find_answer_rows(data, 0, "best", ["false a", "false b"])


    def test_single_base_pairs_file_reconstructs_two_folds_in_memory(self) -> None:
        choices = np.asarray(
            ["best0", "false0", "best1", "false1", "best2", "false2", "best3", "false3"]
        )
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        q_indices = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        positive_rows = np.asarray([0, 2, 4, 6], dtype=np.int64)
        negative_rows = np.asarray([1, 3, 5, 7], dtype=np.int64)
        vector_questions = np.asarray([0, 1, 2, 3], dtype=np.int64)
        vectors = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single_pairs.npz"
            np.savez(
                path,
                vector_q_indices=vector_questions,
                positive_sample_indices=positive_rows,
                negative_sample_indices=negative_rows,
                choices=choices,
                labels=labels,
                q_indices=q_indices,
                v_T=vectors,
            )

            fold0 = load_pair_vector_data(str(path), fold_idx=0, num_folds=2)
            fold1 = load_pair_vector_data(str(path), fold_idx=1, num_folds=2)

        # np.array_split([0,1,2,3], 2) -> [0,1] and [2,3].
        # The selected fold is held out, so training uses the complement.
        np.testing.assert_array_equal(fold0.train_questions, [2, 3])
        np.testing.assert_array_equal(fold1.train_questions, [0, 1])
        self.assertEqual(set(fold0.vectors), {(4, 5), (6, 7)})
        self.assertEqual(set(fold1.vectors), {(0, 1), (2, 3)})
        np.testing.assert_allclose(fold0.shared_mean, [3.5, 0.0])
        np.testing.assert_allclose(fold1.shared_mean, [1.5, 0.0])
        self.assertEqual(fold0.fold_idx, 0)
        self.assertEqual(fold1.fold_idx, 1)

    def test_base_pairs_without_fold_idx_requires_explicit_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single_pairs.npz"
            np.savez(
                path,
                vector_q_indices=np.asarray([0], dtype=np.int64),
                positive_sample_indices=np.asarray([0], dtype=np.int64),
                negative_sample_indices=np.asarray([1], dtype=np.int64),
                choices=np.asarray(["best", "false"]),
                labels=np.asarray([1, 0], dtype=np.int64),
                q_indices=np.asarray([0, 0], dtype=np.int64),
                v_T=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            with self.assertRaises(KeyError):
                load_pair_vector_data(str(path))



if __name__ == "__main__":
    unittest.main()
