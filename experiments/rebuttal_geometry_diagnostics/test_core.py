import numpy as np
import torch

from experiments.rebuttal_geometry_diagnostics.core import (
    GRID, dynamic_segment, equal_budget, split_radial,
)
from experiments.rebuttal_geometry_diagnostics.data import config


def test_radial_tangent_are_orthogonal_and_reconstruct():
    h = torch.tensor([[3., 4., 0.]])
    v = torch.tensor([[1., 2., 3.]])
    radial, tangent = split_radial(v, h)
    torch.testing.assert_close(radial + tangent, v)
    torch.testing.assert_close((radial * tangent).sum(-1), torch.zeros(1), atol=1e-6, rtol=0)


def test_equal_budget_probes_have_same_relative_norm():
    h = torch.tensor([[3., 4., 0.]])
    a = torch.tensor([[1., 2., 3.]])
    b = torch.tensor([[0., 1., 0.]])
    da, db = equal_budget(a, h), equal_budget(b, h)
    torch.testing.assert_close(da.norm(dim=-1), db.norm(dim=-1))
    torch.testing.assert_close(da.norm(dim=-1) / h.norm(dim=-1), torch.tensor([.01]))


def test_adaptive_and_fixed_slerp_grid_are_finite():
    h = torch.randn(5, 8)
    mean, residual = torch.randn(8), torch.randn(8)
    edited, gate, weights = dynamic_segment(h, mean, residual, 3., -.7)
    assert edited.shape == h.shape and gate.shape == (5, 1) and weights.shape == (5, 3)
    assert torch.isfinite(edited).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(5), atol=1e-6, rtol=0)
    assert len(GRID) == 6
    for triple in GRID:
        candidate, _, _ = dynamic_segment(h, mean, residual, 3., -.7, weights=triple)
        assert torch.isfinite(candidate).all()


def test_task_hyperparameters_come_from_benchmark_configs():
    boolq, truthful = config('boolq'), config('truthfulqa')
    assert boolq['weight_kappa'] == 7 and boolq['margin'] == .9
    assert boolq['spherical'] == {'kappa': 20., 'alpha': .5, 'beta': -1.}
    assert truthful['weight_kappa'] == 3 and truthful['margin'] == -.7
    assert truthful['spherical'] == {'kappa': 20., 'alpha': .7, 'beta': -.15}


def test_suffix_scores_and_gradients_match_full_model_in_float32():
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.rebuttal_geometry_diagnostics.core import SuffixEngine
    torch.manual_seed(19)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0., max_position_embeddings=32,
    )).float().eval()
    engine = SuffixEngine(model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    hidden = engine.capture(ids, 0)
    score = engine.scores(hidden, ids, 2, 0, True)
    with torch.no_grad():
        lp = model(ids, use_cache=False).logits[:, 2:-1].log_softmax(-1)
        full = lp.gather(-1, ids[:, 3:, None]).squeeze(-1).mean(-1)
    torch.testing.assert_close(score, full, atol=1e-6, rtol=1e-6)
    leaf = hidden.clone().requires_grad_(True)
    _, gradient = engine.scores(leaf, ids, 2, 0, True, grad=True)
    direction = torch.randn_like(hidden)
    direction[:, :2] = 0
    direction = direction / direction.norm()
    epsilon = 1e-3
    plus = engine.scores(hidden + epsilon * direction, ids, 2, 0, True)
    minus = engine.scores(hidden - epsilon * direction, ids, 2, 0, True)
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic = (gradient * direction).sum().reshape(1)
    torch.testing.assert_close(finite_difference, analytic, atol=3e-4, rtol=3e-3)
    assert gradient[:, -1].abs().max() == 0
