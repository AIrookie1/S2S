from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

import method.l2s_linear.model as linear_model
import method.l2s_slerp.model as slerp_model
from method.S2S_R.predictor import fit_dynamic_mlp


def _toy_l2s_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 4)).astype(np.float32)
    y = rng.normal(size=(12, 4)).astype(np.float32)
    q = np.repeat(np.arange(6), 2)
    return x, y, q


@pytest.mark.parametrize("model_module", [linear_model, slerp_model], ids=["linear", "slerp"])
def test_zero_noise_is_deterministic(model_module):
    x, y, q = _toy_l2s_data(0)
    kwargs = dict(
        hidden_dim=5,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        seed=7,
        target_noise_sigma=0.0,
    )
    a, _, _ = model_module.fit_l2s(x, y, q, np.arange(4), **kwargs)
    b, _, _ = model_module.fit_l2s(x, y, q, np.arange(4), **kwargs)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


@pytest.mark.parametrize("model_module", [linear_model, slerp_model], ids=["linear", "slerp"])
def test_l2s_noise_has_exact_relative_magnitude_and_metadata(model_module):
    x, y, q = _toy_l2s_data(1)
    sigma = 2.0
    train_q = np.arange(4)
    _, _, metrics = model_module.fit_l2s(
        x,
        y,
        q,
        train_q,
        hidden_dim=5,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        seed=7,
        target_noise_sigma=sigma,
        noise_seed=11,
    )
    train_targets = y[np.isin(q, train_q)]
    expected_mse = sigma**2 * np.mean(np.sum(train_targets**2, axis=1)) / y.shape[1]
    assert metrics["target_noise_sigma"] == sigma
    assert metrics["noise_seed"] == 11
    assert np.isclose(metrics["noisy_train_target_mse"], expected_mse, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("model_module", [linear_model, slerp_model], ids=["linear", "slerp"])
def test_l2s_noise_is_sampled_once_before_all_epochs(monkeypatch, model_module):
    x, y, q = _toy_l2s_data(2)
    calls = 0
    original_randn = torch.randn

    def counted_randn(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(model_module.torch, "randn", counted_randn)
    model_module.fit_l2s(
        x,
        y,
        q,
        np.arange(4),
        hidden_dim=5,
        epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        seed=7,
        target_noise_sigma=3.0,
        noise_seed=11,
    )
    assert calls == 1


def test_S2S_R_keeps_v_avg_clean_and_only_corrupts_mlp_targets():
    rng = np.random.default_rng(3)
    h_neg = rng.normal(size=(8, 4)).astype(np.float32)
    clean_vectors = rng.normal(size=(8, 4)).astype(np.float32)
    h_pos = h_neg + clean_vectors
    q = np.arange(8)
    mean_mask = np.array([True, True, True, True, False, False, False, False])
    mlp_mask = ~mean_mask
    sigma = 5.0

    state, _, metrics = fit_dynamic_mlp(
        h_neg,
        h_pos,
        q,
        mean_mask,
        mlp_mask,
        mlp_hidden=5,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        seed=7,
        target_noise_sigma=sigma,
        noise_seed=11,
    )

    expected_v_avg = clean_vectors[mean_mask].mean(axis=0)
    assert np.allclose(state["v_avg"].numpy(), expected_v_avg, rtol=1e-6, atol=1e-6)

    mlp_targets = clean_vectors[mlp_mask]
    expected_mse = sigma**2 * np.mean(np.sum(mlp_targets**2, axis=1)) / clean_vectors.shape[1]
    assert np.isclose(metrics["noisy_train_target_mse"], expected_mse, rtol=1e-5, atol=1e-6)


def test_experiment_pins_inference_noise_off_and_uses_requested_default_levels():
    run_text = (Path(__file__).parent / "fig5_run.sh").read_text(encoding="utf-8")
    assert 'NOISE_LEVELS="${NOISE_LEVELS:-1.0 2.0 3.0 4.0 5.0}"' in run_text
    assert run_text.count("--residual_noise_sigma 0.0") == 3
    assert "target_noise_sampling" in run_text
