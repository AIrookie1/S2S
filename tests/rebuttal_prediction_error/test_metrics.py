import numpy as np
import torch

from experiments.rebuttal_prediction_error.rebuttal_prediction_error_run import geometry


def test_direction_and_magnitude_are_separate():
    target = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    pred = torch.tensor([[1., 0.], [2., 0.], [0., 1.]])
    result = geometry(pred, target, torch.zeros(2))
    np.testing.assert_allclose(result['angle_deg'], [0, 0, 90], atol=1e-10)
    np.testing.assert_allclose(result['norm_abs_error'], [0, 1, 0])
    np.testing.assert_allclose(result['norm_rel_error_pct'], [0, 100, 0])
    np.testing.assert_allclose(result['vector_l2_error'], [0, 1, np.sqrt(2)])


def test_undefined_direction_and_zero_target_norm():
    result = geometry(torch.tensor([[0., 0.], [1., 0.]]),
                      torch.tensor([[1., 0.], [0., 0.]]), torch.zeros(2))
    assert np.isnan(result['angle_deg']).all()
    assert result['norm_rel_error_pct'][0] == 100
    assert np.isnan(result['norm_rel_error_pct'][1])


def test_residual_angle_uses_mean_subtraction():
    result = geometry(torch.tensor([[2., 0.]]), torch.tensor([[1., 1.]]),
                      torch.tensor([1., 0.]))
    np.testing.assert_allclose(result['angle_deg'], [45])
    np.testing.assert_allclose(result['residual_angle_deg'], [90])


def test_official_boolq_and_copa_explicitly_load_validation(monkeypatch):
    from experiments.rebuttal_prediction_error import extract_official as extraction
    calls = []

    def fake_load(name, dataset, split):
        calls.append((name, dataset, split))
        if dataset == 'boolq':
            return [{'passage': 'P', 'question': 'Q', 'label': 1}]
        return [{'premise': 'P', 'question': 'cause', 'choice1': 'C1',
                 'choice2': 'C2', 'label': 0}]

    monkeypatch.setattr(extraction, 'load_dataset', fake_load)
    prompts, labels, q = extraction.official_data('boolq', None)
    assert prompts[0].endswith(' no') and prompts[1].endswith(' yes')
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_array_equal(q, [0, 0])
    prompts, labels, q = extraction.official_data('copa', None)
    assert prompts[0].endswith(' C1') and prompts[1].endswith(' C2')
    np.testing.assert_array_equal(labels, [1, 0])
    assert calls == [('super_glue', 'boolq', 'validation'),
                     ('super_glue', 'copa', 'validation')]
