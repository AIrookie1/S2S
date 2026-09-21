from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from method.dyn3d_sga.dynSVD import load_feature_pairs
from method.dynlearnedslerp.model import (
    ResidualParamsNet as LearnedSLERPParamsNet,
    ResidualRouter as LearnedSLERPRouter,
    predict_direction as predict_learned_slerp_direction,
    residual_params_true,
)
from method.l2s.model import load_l2s_pairs
from method.l2s_slerp.steering import l2s_prediction_to_slerp
from method.sadi.hidden import rank_hidden_units, tune_hidden_sadi
from method.sadi.steering import sadi_hidden_intervention

MODULE_PATH = Path(__file__).resolve().parent / "evaluate.py"
spec = importlib.util.spec_from_file_location("cosine_similarity_evaluate", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)



def test_dyn3d_cartesian_pairs_exactly_match_l2s_raw_pairs(tmp_path):
    # Include answer-level fields whose mean-positive construction would differ,
    # plus the exact raw Cartesian pair fields used by L2S.
    activations = np.asarray(
        [
            [2.0, 0.0],   # q0 positive 1
            [0.0, 2.0],   # q0 positive 2
            [-1.0, 0.0],  # q0 negative 1
            [0.0, -1.0],  # q0 negative 2
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    q_indices = np.asarray([0, 0, 0, 0], dtype=np.int64)
    choice_indices = np.arange(4, dtype=np.int64)

    z_pos_raw = np.asarray(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    z_neg_raw = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    v_t = z_pos_raw - z_neg_raw
    vector_q = np.zeros(4, dtype=np.int64)

    feature = tmp_path / "pairs.npz"
    np.savez_compressed(
        feature,
        activations=activations,
        labels=labels,
        q_indices=q_indices,
        choice_indices=choice_indices,
        z_pos_raw=z_pos_raw,
        z_neg_raw=z_neg_raw,
        v_T=v_t,
        vector_q_indices=vector_q,
    )

    dyn_neg, dyn_pos, dyn_q, meta = load_feature_pairs(
        feature, positive_mode="cartesian"
    )
    l2s_neg, l2s_v, l2s_q, _ = load_l2s_pairs(feature)

    np.testing.assert_allclose(dyn_neg, l2s_neg, atol=0.0)
    np.testing.assert_allclose(dyn_pos - dyn_neg, l2s_v, atol=0.0)
    np.testing.assert_array_equal(dyn_q, l2s_q)

    # Ensure the old mean-positive target really would be different.
    mean_neg, mean_pos, _, _ = load_feature_pairs(
        feature, positive_mode="mean"
    )
    assert len(mean_neg) == 2
    assert len(dyn_neg) == 4
    assert not np.array_equal(mean_pos[0], dyn_pos[0])

def test_cosine_summary_identity_and_opposite():
    target = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    pred = np.asarray([[3.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    stats = mod._summarize_cosine(
        pred, target, np.asarray([10, 11], dtype=np.int64)
    )
    assert stats["num_pairs"] == 2
    assert stats["num_negatives"] == 2
    assert np.isclose(stats["cosine"], 0.0, atol=1e-7)
    assert np.isclose(stats["min"], -1.0, atol=1e-7)
    assert np.isclose(stats["max"], 1.0, atol=1e-7)


def test_cosine_uses_mean_over_all_pairs_not_nearest_gt():
    # Negative 100 has two correct GTs with cosine 0 and 1.
    # Negative 200 has one GT with cosine 1.
    # All-pairs mean must therefore be (0 + 1 + 1) / 3 = 2/3,
    # not nearest-GT mean (1 + 1) / 2 = 1.
    pred = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    negative_ids = np.asarray([100, 100, 200], dtype=np.int64)

    stats = mod._summarize_cosine(pred, target, negative_ids)
    assert stats["num_pairs"] == 3
    assert stats["num_negatives"] == 2
    assert np.isclose(stats["cosine"], 2.0 / 3.0, atol=1e-7)


def test_standard_mean_is_full_outer_train_pair_mean():
    v_t = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [-5.0, 0.0],
        ],
        dtype=np.float32,
    )
    q = np.asarray([0, 0, 1, 2], dtype=np.int64)
    train_q = np.asarray([0, 1], dtype=np.int64)
    mask = mod._mask_for_questions(q, train_q, label="synthetic train")
    avg = v_t[mask].mean(axis=0)
    np.testing.assert_allclose(avg, np.asarray([2.0 / 3.0, 2.0 / 3.0], dtype=np.float32))


def test_dyn3d_prediction_is_direct_mean_residual_slerp():
    # This test verifies the new cosine-only Dyn3D definition:
    # v_pred = SLERP(norm(v_avg), residual, gamma / (beta + gamma)).
    # The returned vector must be used directly, with no candidate-z_neg step.
    z_neg = np.asarray([[3.0, 4.0]], dtype=np.float32)

    class Router(torch.nn.Module):
        def forward(self, z):
            return torch.ones((len(z), 1), dtype=z.dtype, device=z.device)

    def norm_fn(x, eps=1e-8):
        return torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=eps)

    def weights(a, b, c, kappa, temperature):
        shape = a.shape[:-1] + (1,)
        alpha = torch.full(shape, 0.2, dtype=a.dtype, device=a.device)
        beta = torch.full(shape, 0.2, dtype=a.dtype, device=a.device)
        gamma = torch.full(shape, 0.6, dtype=a.dtype, device=a.device)
        return alpha, beta, gamma, {}

    captured = {}

    def direct_slerp(x, y, t, eps, threshold):
        captured["x"] = x.detach().clone()
        captured["y"] = y.detach().clone()
        captured["t"] = t.detach().clone()
        # Lightweight interpolation for dependency injection. Production uses
        # the repository's actual spherical slerp implementation.
        return (1.0 - t) * x + t * y

    pred = mod.predict_dyn3d_vectors(
        z_neg,
        avg=torch.asarray([1.0, 0.0]),
        basis=torch.asarray([[0.0, 1.0]]),
        router=Router(),
        norm_fn=norm_fn,
        adaptive_weights_fn=weights,
        slerp_fn=direct_slerp,
        weight_kappa=3.0,
        device=torch.device("cpu"),
        batch_size=8,
    )

    np.testing.assert_allclose(
        captured["x"].cpu().numpy(),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        captured["y"].cpu().numpy(),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        captured["t"].cpu().numpy(),
        np.asarray([[0.75]], dtype=np.float32),
        atol=1e-7,
    )
    # Direct interpolation output: [0.25, 0.75]. If the old
    # candidate-z_neg rule were still present, this would be very different.
    np.testing.assert_allclose(
        pred,
        np.asarray([[0.25, 0.75]], dtype=np.float32),
        atol=1e-7,
    )




def test_l2s_slerp_uses_l2s_prediction_minus_vavg_and_adaptive_t():
    # L2S predicts [1, 2], while v_avg is [1, 0], so residual direction is
    # exactly [0, 1].  t must be generated from the adaptive three-way weights.
    z_neg = torch.asarray([[1.0, 1.0]], dtype=torch.float32)
    direct_v = torch.asarray([[1.0, 2.0]], dtype=torch.float32)
    v_avg = torch.asarray([1.0, 0.0], dtype=torch.float32)

    out, info = l2s_prediction_to_slerp(
        z_neg, direct_v, v_avg, weight_kappa=3.0, return_info=True
    )
    expected_t = info["w_residual"] / (info["w_mean"] + info["w_residual"])
    np.testing.assert_allclose(
        info["t_adaptive"].detach().cpu().numpy(),
        expected_t.detach().cpu().numpy(),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        torch.linalg.vector_norm(out, dim=-1).detach().cpu().numpy(),
        torch.linalg.vector_norm(direct_v, dim=-1).detach().cpu().numpy(),
        atol=1e-6,
    )


def test_dynlearnedslerp_predicts_learned_slerp_position():
    # Rank=1 softmax Router always produces the single [0, 1] basis direction.
    router = LearnedSLERPRouter(
        input_dim=2,
        rank=1,
        hidden_dim=2,
        top_k=1,
    )
    params = LearnedSLERPParamsNet(
        input_dim=2,
        hidden_dim=2,
    )
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
        for parameter in params.parameters():
            parameter.zero_()
        # sigmoid(0) = 0.5
        params.fc2.bias.zero_()

    pred, info = predict_learned_slerp_direction(
        torch.asarray([[3.0, 4.0]]),
        torch.asarray([1.0, 0.0]),
        torch.asarray([[0.0, 1.0]]),
        router,
        params,
    )

    expected = np.asarray(
        [[2.0 ** -0.5, 2.0 ** -0.5]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        pred.detach().cpu().numpy(),
        expected,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        info["residual_direction"].detach().cpu().numpy(),
        [[0.0, 1.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        info["slerp_t"].detach().cpu().numpy(),
        [0.5],
        atol=1e-7,
    )


def test_dynlearnedslerp_true_t_reconstructs_raw_vector_direction():
    mean = torch.asarray([[2.0, 0.0]])
    residual = torch.asarray([[0.0, 1.0]])
    t = residual_params_true(mean, residual)

    # The target vector is mean + residual = [2, 1].
    from method.dynlearnedslerp.model import slerp_unit

    reconstructed = slerp_unit(
        torch.nn.functional.normalize(mean, dim=-1),
        torch.nn.functional.normalize(residual, dim=-1),
        t,
    )
    expected = torch.nn.functional.normalize(
        mean + residual,
        dim=-1,
    )
    torch.testing.assert_close(
        reconstructed,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

def test_dynmlp_prediction_is_direct_mean_residual_slerp():
    z_neg = np.asarray([[3.0, 4.0]], dtype=np.float32)

    class ResidualMLP(torch.nn.Module):
        def forward(self, z):
            out = torch.zeros_like(z)
            out[:, 1] = 1.0
            return out

    def norm_fn(x, eps=1e-8):
        return torch.nn.functional.normalize(x.float(), p=2, dim=-1, eps=eps)

    def weights(a, b, c, kappa, temperature):
        shape = a.shape[:-1] + (1,)
        alpha = torch.full(shape, 0.2, dtype=a.dtype, device=a.device)
        beta = torch.full(shape, 0.2, dtype=a.dtype, device=a.device)
        gamma = torch.full(shape, 0.6, dtype=a.dtype, device=a.device)
        return alpha, beta, gamma, {}

    captured = {}

    def direct_slerp(x, y, t, eps, threshold):
        captured["x"] = x.detach().clone()
        captured["y"] = y.detach().clone()
        captured["t"] = t.detach().clone()
        return (1.0 - t) * x + t * y

    pred = mod.predict_dynmlp_vectors(
        z_neg,
        avg=torch.asarray([1.0, 0.0]),
        model=ResidualMLP(),
        norm_fn=norm_fn,
        adaptive_weights_fn=weights,
        slerp_fn=direct_slerp,
        weight_kappa=3.0,
        device=torch.device("cpu"),
        batch_size=8,
    )

    np.testing.assert_allclose(
        captured["x"].cpu().numpy(),
        [[1.0, 0.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        captured["y"].cpu().numpy(),
        [[0.0, 1.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        captured["t"].cpu().numpy(),
        [[0.75]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        pred,
        [[0.25, 0.75]],
        atol=1e-7,
    )

def test_dyn3d_runtime_reuses_existing_source_without_lm_eval_dependencies(tmp_path):
    d, hidden, rank = 3, 4, 2
    ckpt = tmp_path / "dyn.npz"
    np.savez_compressed(
        ckpt,
        hidden_dim=np.asarray(d, np.int64),
        rank=np.asarray(rank, np.int64),
        router_hidden=np.asarray(hidden, np.int64),
        router_top_k=np.asarray(rank, np.int64),
        v_avg=np.asarray([1.0, 0.2, 0.1], np.float32),
        basis=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        **{
            "router_fc1.weight": np.zeros((hidden, d), np.float32),
            "router_fc1.bias": np.zeros(hidden, np.float32),
            "router_fc2.weight": np.zeros((rank, hidden), np.float32),
            "router_fc2.bias": np.zeros(rank, np.float32),
        },
    )
    data, avg, basis, router, norm_fn, weights_fn, slerp_fn = mod.load_dyn3d_runtime(
        ckpt, torch.device("cpu")
    )
    try:
        assert avg.shape == (d,)
        assert basis.shape == (rank, d)
        x = norm_fn(torch.asarray([[2.0, 1.0, 0.5]]), 1e-8)
        coeff = router(x)
        residual = mod.normalized_basis_combination(coeff, basis)
        b = norm_fn(avg, 1e-8).view(1, -1)
        alpha, beta, gamma, _ = weights_fn(
            x, b, residual, kappa=3.0, temperature=1.0
        )
        t = gamma / (beta + gamma).clamp_min(1e-8)
        out = slerp_fn(b, residual, t, 1e-8, 1e-4)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
    finally:
        data.close()


def test_sadi_hidden_prediction_is_dynamic_mlp_delta():
    mlp_z_neg = np.asarray(
        [[2.0, -3.0, 4.0], [-1.0, 5.0, 7.0]],
        dtype=np.float32,
    )
    ranked = np.asarray([1, 0, 2], dtype=np.int64)
    pred = mod.predict_hidden_vectors(
        mlp_z_neg,
        ranked,
        count=2,
        strength=5.0,
    )
    expected = np.asarray(
        [[8.0, -12.0, 0.0], [-4.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pred, expected, atol=1e-7)


def test_sadi_ranking_uses_only_requested_geometry_questions():
    mlp_v_t = np.asarray(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [0.0, 100.0],
        ],
        dtype=np.float32,
    )
    q = np.asarray([0, 0, 1], dtype=np.int64)
    ranked, mean_difference = rank_hidden_units(
        mlp_v_t,
        q,
        np.asarray([0], dtype=np.int64),
    )
    assert ranked.tolist() == [0, 1]
    np.testing.assert_allclose(mean_difference, [2.5, 0.0], atol=1e-7)


def test_sadi_validation_tuning_uses_pairwise_mean_and_breaks_strength_tie():
    # Both rows belong to the same negative, but pairwise-mean validation must
    # score both GT rows independently instead of selecting the nearest one.
    mlp_z_neg = np.asarray(
        [[1.0, 10.0], [1.0, -10.0]],
        dtype=np.float32,
    )
    decoder_v_t = np.asarray(
        [[1.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    negative_ids = np.asarray([77, 77], dtype=np.int64)
    q = np.asarray([5, 5], dtype=np.int64)
    ranked = np.asarray([0, 1], dtype=np.int64)
    best, history = tune_hidden_sadi(
        mlp_z_neg,
        decoder_v_t,
        negative_ids,
        q,
        np.asarray([5], dtype=np.int64),
        ranked,
        counts=[1, 2],
        strengths=[5.0, 10.0],
    )
    assert best["count"] == 1
    assert best["strength"] == 5.0
    assert np.isclose(best["val_cosine"], 1.0, atol=1e-7)
    assert len(history) == 4


def test_sadi_runtime_hook_modifies_selected_target_layer_mlp_once():
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Identity()

    class FakeCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeCore()

    model = FakeModel()
    x = torch.asarray([[[1.0, 2.0, 3.0]]])
    with sadi_hidden_intervention(
        model,
        layer_idx=0,
        hidden_indices=[1],
        strength=5.0,
        apply_once=True,
    ):
        first = model.model.layers[0].mlp(x)
        second = model.model.layers[0].mlp(x)
    torch.testing.assert_close(first, torch.asarray([[[1.0, 10.0, 3.0]]]))
    torch.testing.assert_close(second, x)
