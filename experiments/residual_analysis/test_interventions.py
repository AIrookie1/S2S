"""Regression tests for the residual-analysis intervention definitions.

The production evaluator depends on optional experiment packages (baukit,
transformers, TruthfulQA).  These tests exercise only the pure geometry/helper
logic, so minimal import stubs are installed when those packages are absent.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _install_optional_import_stubs() -> None:
    try:
        import baukit  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("baukit")
        module.TraceDict = object
        sys.modules["baukit"] = module

    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("datasets")
        module.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = module

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("transformers")
        module.AutoModelForCausalLM = object
        module.AutoTokenizer = object
        sys.modules["transformers"] = module

    try:
        from truthfulqa.models import MC_calcs as _mc_calcs  # noqa: F401
        from truthfulqa.presets import preset_map as _preset_map  # noqa: F401
        from truthfulqa.utilities import format_best as _format_best  # noqa: F401
    except ModuleNotFoundError:
        package = sys.modules.get("truthfulqa", types.ModuleType("truthfulqa"))
        models = types.ModuleType("truthfulqa.models")
        models.MC_calcs = lambda *args, **kwargs: None
        models.set_columns = lambda *args, **kwargs: None
        presets = types.ModuleType("truthfulqa.presets")
        presets.preset_map = {"qa": ""}
        utilities = types.ModuleType("truthfulqa.utilities")
        utilities.format_best = lambda value: value
        utilities.format_prompt_with_answer_strings = (
            lambda question, choice, preset, format="general": f"Q: {question} A: {choice}"
        )
        utilities.split_multi_answer = lambda value: str(value).split(";")
        sys.modules["truthfulqa"] = package
        sys.modules["truthfulqa.models"] = models
        sys.modules["truthfulqa.presets"] = presets
        sys.modules["truthfulqa.utilities"] = utilities


_install_optional_import_stubs()

from experiments.residual_analysis.evaluate import (  # noqa: E402
    CandidateRecord,
    SphericalPrototype,
    SphericalPrototypeBank,
    _nearest_positive_target,
    compute_negative_geometry,
    cross_category_cosine_nearest_indices,
    _make_cross_category_cosine_nearest_map,
    get_dynamic_residual_intervention,
    get_static_2d_residual_intervention,
    load_spherical_prototype_bank,
    parse_args,
    validate_spherical_coverage,
)
from method.dyn2d.steering import (  # noqa: E402
    get_2d_dynamic_intervention,
    slerp as dyn2d_slerp,
)
from method.dyn3d_sga.dyn_mlp import DirectVectorMLP  # noqa: E402
from method.dyn3d_sga.evaluate_3d_dyn import (  # noqa: E402
    adaptive_3way_weights,
    norm,
)


@pytest.mark.parametrize("condition", ["shuffled_residual", "oracle_residual"])
def test_override_residual_recomputes_adaptive_weights(condition: str) -> None:
    avg = torch.tensor([1.0, 0.0, 0.0])
    predictor = DirectVectorMLP(3, 4, 3)
    for parameter in predictor.parameters():
        torch.nn.init.zeros_(parameter)

    hidden = torch.tensor(
        [[[0.2, 1.0, 0.1], [0.1, 0.3, 1.0]]],
        dtype=torch.float32,
    )
    override = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    stats: dict[str, float] = {}
    hook = get_dynamic_residual_intervention(
        avg=avg,
        predictor=predictor,
        condition=condition,
        residual_override=override,
        weight_kappa=3.0,
        margin=-0.7,
        eps=1e-8,
        small_angle_threshold=1e-4,
        stats=stats,
    )
    output = hook(hidden.clone(), start_idx=0)
    assert torch.isfinite(output).all()

    a = norm(hidden)
    b = norm(avg).view(1, 1, -1).expand_as(a)
    c = torch.as_tensor(override).view(1, 1, -1).expand_as(a)
    alpha, beta, gamma, _ = adaptive_3way_weights(
        a, b, c, kappa=3.0, temperature=1.0
    )
    count = float(stats["weight_count"])
    assert np.isclose(float(stats["alpha_sum"]) / count, float(alpha.mean()))
    assert np.isclose(float(stats["beta_sum"]) / count, float(beta.mean()))
    assert np.isclose(float(stats["gamma_sum"]) / count, float(gamma.mean()))


def test_cross_fitted_spherical_prototype_bank(tmp_path: Path) -> None:
    for fold, questions in ((0, [0, 1]), (1, [2, 3])):
        np.savez(
            tmp_path / f"prototype_fold{fold}.npz",
            mu_T=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            mu_H=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            test_q_indices=np.asarray(questions, dtype=np.int64),
            fold_idx=np.asarray(fold, dtype=np.int64),
        )

    bank = load_spherical_prototype_bank(
        [
            str(tmp_path / "prototype_fold0.npz"),
            str(tmp_path / "prototype_fold1.npz"),
        ],
        expected_dim=3,
    )
    validate_spherical_coverage(bank, {0, 1, 2, 3})
    assert bank.for_question(0).fold_idx == 0
    assert bank.for_question(3).fold_idx == 1


def _constant_predictor(vector: torch.Tensor) -> DirectVectorMLP:
    predictor = DirectVectorMLP(int(vector.numel()), 4, int(vector.numel()))
    with torch.no_grad():
        predictor.fc1.weight.zero_()
        predictor.fc1.bias.zero_()
        predictor.fc2.weight.zero_()
        predictor.fc2.bias.copy_(vector.float())
    return predictor


def test_s2s_l_static_override_uses_recipient_predicted_norm() -> None:
    avg = torch.tensor([1.0, 0.0, 0.0])
    # Constant recipient prediction with ||v_hat|| = 2.  The override below is
    # a direction only and must not carry any donor magnitude into the edit.
    predictor = _constant_predictor(torch.tensor([0.0, 0.0, 2.0]))
    hidden = torch.tensor(
        [[[0.3, 0.1, 0.2], [0.1, 0.4, 0.5]]],
        dtype=torch.float32,
    )
    override = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    stats: dict[str, float] = {}
    hook = get_static_2d_residual_intervention(
        avg=avg,
        predictor=predictor,
        condition="shuffled_residual",
        residual_override=override,
        strength=2.0,
        slerp_t=0.25,
        eps=1e-8,
        small_angle_threshold=1e-4,
        stats=stats,
    )
    output = hook(hidden.clone(), start_idx=0)

    mean = norm(avg).view(1, -1)
    donor_direction = torch.as_tensor(override).view(1, -1)
    direction = dyn2d_slerp(mean, donor_direction, 0.25)
    expected = hidden + 2.0 * 2.0 * direction.view(1, 1, -1)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)
    assert np.isclose(stats["predicted_norm_sum"] / stats["prediction_count"], 2.0)
    assert np.isclose(stats["applied_norm_sum"] / stats["prediction_count"], 4.0)
    assert np.isclose(stats["slerp_t_sum"] / stats["slerp_t_count"], 0.25)


def test_s2s_l_static_predicted_condition_matches_released_dyn2d_hook() -> None:
    avg = torch.tensor([1.0, 0.0, 0.0])
    predictor = _constant_predictor(torch.tensor([0.2, -0.1, 1.7]))
    hidden = torch.tensor(
        [[[0.3, 0.5, 0.1], [0.4, -0.2, 0.8]]],
        dtype=torch.float32,
    )
    reference = get_2d_dynamic_intervention(
        avg,
        predictor,
        strength=1.3,
        slerp_t=0.35,
        eps=1e-8,
        small_angle_threshold=1e-4,
        method="S2S_L_Static",
    )
    analysis = get_static_2d_residual_intervention(
        avg=avg,
        predictor=predictor,
        condition="predicted_residual",
        residual_override=None,
        strength=1.3,
        slerp_t=0.35,
        eps=1e-8,
        small_angle_threshold=1e-4,
    )
    reference_output = reference(hidden.clone(), start_idx=0)
    analysis_output = analysis(hidden.clone(), start_idx=0)
    assert torch.allclose(analysis_output, reference_output, atol=1e-6, rtol=1e-6)


def test_s2s_l_static_invalid_oracle_falls_back_to_mean_direction() -> None:
    avg = torch.tensor([1.0, 0.0, 0.0])
    predictor = _constant_predictor(torch.tensor([0.0, 0.0, 2.0]))
    hidden = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    hook = get_static_2d_residual_intervention(
        avg=avg,
        predictor=predictor,
        condition="oracle_residual",
        residual_override=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        oracle_override_valid=False,
        strength=1.5,
        slerp_t=0.8,
        eps=1e-8,
        small_angle_threshold=1e-4,
    )
    output = hook(hidden.clone(), start_idx=0)
    expected = hidden + 1.5 * 2.0 * norm(avg).view(1, 1, -1)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_panel_c_s2s_l_static_uses_real_additive_geometry() -> None:
    avg = torch.tensor([1.0, 0.0, 0.0])
    hidden = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    positive = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    residual = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    record = CandidateRecord(
        candidate_id="q0:F0",
        q_idx=0,
        row_idx=0,
        category="test",
        question="q",
        label=0,
        answer_index=0,
        answer="a",
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        start_idx=0,
        hidden=hidden,
        positive_target=positive,
        positive_target_kind="nearest_truthful_answer",
        positive_target_answer_index=1,
        positive_target_cosine=0.9,
        predicted_residual=residual,
        predicted_vector_norm=2.0,
        predicted_residual_raw_norm=1.0,
        predicted_valid=True,
        oracle_residual=residual,
        oracle_residual_raw_norm=1.0,
        oracle_valid=True,
    )
    hidden_unit = hidden / np.linalg.norm(hidden)
    bank = SphericalPrototypeBank(
        by_question={
            0: SphericalPrototype(
                fold_idx=0,
                mu_T=hidden_unit.astype(np.float32),
                mu_H=(-hidden_unit).astype(np.float32),
                source_path="synthetic",
            )
        },
        source_paths=["synthetic"],
    )
    rows = compute_negative_geometry(
        [record],
        avg=avg,
        steering_method="S2S_L_Static",
        strength=1.5,
        slerp_t=0.25,
        spherical_bank=bank,
        spherical_kappa=20.0,
        spherical_alpha=0.7,
        spherical_beta=0.9,
        weight_kappa=3.0,
        margin=-0.7,
        eps=1e-8,
        small_angle_threshold=1e-4,
    )
    assert len(rows) == 1

    direction = dyn2d_slerp(
        norm(avg).view(1, -1),
        torch.as_tensor(residual).view(1, -1),
        0.25,
    )[0]
    edited = torch.as_tensor(hidden) + 1.5 * 2.0 * direction
    edited = norm(edited)
    target = norm(torch.as_tensor(positive))
    expected_theta_pred = float(
        torch.rad2deg(torch.acos(torch.dot(edited, target).clamp(-1.0, 1.0)))
    )
    assert np.isclose(rows[0]["theta_predicted_deg"], expected_theta_pred, atol=1e-5)

    # S2S-L-Static must use a linear additive mean-only baseline, not the
    # spherical prototype bank: h_mean = h + strength * v_avg.
    mean_edited = norm(torch.as_tensor(hidden) + 1.5 * avg)
    expected_theta_mean = float(
        torch.rad2deg(torch.acos(torch.dot(mean_edited, target).clamp(-1.0, 1.0)))
    )
    assert np.isclose(rows[0]["theta_mean_deg"], expected_theta_mean, atol=1e-5)
    assert rows[0]["mean_baseline_kind"] == "linear_additive_v_avg"
    assert rows[0]["spherical_mean_gate"] == ""
    assert rows[0]["spherical_mean_fold"] == ""
    assert np.isclose(rows[0]["linear_mean_strength"], 1.5)
    assert np.isnan(rows[0]["alpha"])
    assert np.isclose(rows[0]["strength"], 1.5)
    assert np.isclose(rows[0]["slerp_t"], 0.25)



def test_panel_c_s2s_l_static_does_not_require_spherical_bank() -> None:
    avg = torch.tensor([0.5, 0.0, 0.0])
    hidden = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    positive = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    residual = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    record = CandidateRecord(
        candidate_id="q0:F0", q_idx=0, row_idx=0, category="test", question="q",
        label=0, answer_index=0, answer="a", input_ids=torch.tensor([[1, 2]]),
        start_idx=0, hidden=hidden, positive_target=positive,
        positive_target_kind="nearest_truthful_answer",
        positive_target_answer_index=0, positive_target_cosine=0.7,
        predicted_residual=residual, predicted_vector_norm=1.0,
        predicted_residual_raw_norm=1.0, predicted_valid=True,
        oracle_residual=residual, oracle_residual_raw_norm=1.0, oracle_valid=True,
    )
    rows = compute_negative_geometry(
        [record], avg=avg, steering_method="S2S_L_Static", strength=2.0,
        slerp_t=0.2, spherical_bank=None, spherical_kappa=20.0,
        spherical_alpha=0.7, spherical_beta=-0.15, weight_kappa=3.0,
        margin=-0.7, eps=1e-8, small_angle_threshold=1e-4,
    )
    assert rows[0]["mean_baseline_kind"] == "linear_additive_v_avg"



def test_s2s_l_parser_does_not_require_spherical_prototypes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "llama3.1-8B-Instruct",
            "--checkpoint_path",
            "dummy.npz",
            "--steering_method",
            "S2S_L_Static",
        ],
    )
    args = parse_args()
    assert args.steering_method == "S2S_L_Static"
    assert args.spherical_prototype_paths is None


def test_s2s_r_parser_still_requires_spherical_prototypes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "llama3.1-8B-Instruct",
            "--checkpoint_path",
            "dummy.npz",
            "--steering_method",
            "S2S_R",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_nearest_positive_target_uses_maximum_cosine() -> None:
    hidden = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    positives = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target, idx, cosine = _nearest_positive_target(hidden, positives, 1e-8)
    assert idx == 1
    assert np.allclose(target, positives[1])
    assert np.isclose(cosine, 1.0)


def test_cross_category_cosine_nearest_indices_selects_best_other_category() -> None:
    residuals = np.asarray([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [-1.0, 0.0],
    ], dtype=np.float32)
    residuals /= np.linalg.norm(residuals, axis=1, keepdims=True)
    categories = np.asarray(["A", "A", "B", "C"], dtype=object)
    q = np.asarray([0, 1, 2, 3], dtype=np.int64)
    donors, cos = cross_category_cosine_nearest_indices(residuals, categories, q)
    # A rows must choose B row 2 (the closest allowed direction), not A row 1.
    assert donors[0] == 2
    assert donors[1] == 2
    # B row 2 should choose A row 1, which is closer than A row 0.
    assert donors[2] == 1
    assert np.all(np.isfinite(cos))

