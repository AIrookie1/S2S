"""Guard against cross-fold selection leakage and invalid cached experiment data."""
import copy
import unittest

from experiments.rebuttal_R1W2_model_scaling.retune_spherical import (
    expanded_grid, select, validate_report, validate_split, weighted,
)


class RetuneProtocolTests(unittest.TestCase):
    def test_each_fold_is_valid_but_pooled_dev_leaks_into_test(self):
        folds = [dict(geometry=[0], predictor=[1], dev=[2], test=[3, 4, 5]),
                 dict(geometry=[3], predictor=[4], dev=[5], test=[0, 1, 2])]
        for fold in folds:
            validate_split(fold)
        self.assertTrue(set(folds[0]['test']) & set(folds[1]['dev']))
        bad = copy.deepcopy(folds[0])
        bad['dev'].append(3)
        with self.assertRaises(ValueError):
            validate_split(bad)

    def test_cache_cannot_cross_fold_or_hide_duplicates_or_wrong_metrics(self):
        params = dict(kappa=50., alpha=.6, beta=.1)
        report = dict(parameters=params, metrics=dict(MC1=.5, MC2=.5, MC3=.5),
                      details=[dict(question_id=i, MC1=v, MC2=v, MC3=v)
                               for i, v in [(1, 0.), (2, 1.)]])
        validate_report(report, [1, 2], params)
        with self.assertRaises(ValueError):
            validate_report(report, [3, 4], params)
        with self.assertRaises(ValueError):
            validate_report(report, [1, 2], dict(params, alpha=.9))
        bad = copy.deepcopy(report)
        bad['details'][1]['question_id'] = 1
        with self.assertRaises(ValueError):
            validate_report(bad, [1, 2], params)
        bad = copy.deepcopy(report)
        bad['metrics']['MC1'] = 1.
        with self.assertRaises(ValueError):
            validate_report(bad, [1, 2], params)

    def test_selection_uses_mc1_then_mc2_and_stable_order(self):
        cells = [dict(parameters=i, metrics=dict(MC1=a, MC2=b))
                 for i, a, b in [(0, .2, .9), (1, .3, .4), (2, .3, .5), (3, .3, .5)]]
        self.assertEqual(select(cells)['parameters'], 2)
        grid = expanded_grid()
        self.assertEqual(len(grid), 85)
        self.assertEqual(len({tuple(sorted(p.items())) for p in grid}), len(grid))
        self.assertTrue(any(p['alpha'] == 0 for p in grid))

    def test_weighted_heldout_coverage_rejects_overlap(self):
        reports = [dict(question_ids=list(range(408)), metrics=dict(MC1=1., MC2=1., MC3=1.)),
                   dict(question_ids=list(range(408, 817)), metrics=dict(MC1=0., MC2=0., MC3=0.))]
        self.assertAlmostEqual(weighted(reports)['MC1'], 408 / 817)
        reports[1]['question_ids'][0] = 0
        with self.assertRaises(ValueError):
            weighted(reports)


if __name__ == '__main__':
    unittest.main()
