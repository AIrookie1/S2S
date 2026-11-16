"""CPU checks for the mean-only adaptive-weight and margin-gate control."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.component_ablation import evaluate


@pytest.fixture(autouse=True)
def forbid_predictor(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("mean-only control must not call predict_residual")

    monkeypatch.setattr(evaluate, "predict_residual", fail)


def test_mean_only_weights_and_rotation_have_analytic_values():
    avg = torch.tensor([1.0, 0.0, 0.0])
    hidden = torch.tensor([[[0.0, 2.0, 0.0]]])
    stats = {}
    output = evaluate.get_ablation_intervention(
        avg, None, "vavg_aw_gate", weight_kappa=3.0, margin=-0.7, stats=stats
    )(hidden)

    # For X perpendicular to V, the graph's incoming centralities are
    # (2q, 1.5-q, 1.5-q), q=1/(1+exp(kappa)). Merge the last two nodes.
    q = 1.0 / (1.0 + math.exp(3.0))
    alpha = 1.0 / (1.0 + 2.0 * math.exp(3.0 * (1.5 - 3.0 * q)))
    beta = 1.0 - alpha
    expected = torch.tensor([[[2 * math.sin(beta * math.pi / 2),
                               2 * math.cos(beta * math.pi / 2), 0.0]]])
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(hidden, torch.tensor([[[0.0, 2.0, 0.0]]]))
    assert stats["alpha_sum"] == pytest.approx(alpha)
    assert stats["beta_sum"] == pytest.approx(beta)
    assert stats["gamma_sum"] == 0.0
    assert stats["total"] == stats["steered"] == stats["weight_count"] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gate_selects_tokens_and_preserves_tuple_prefix_norm_and_dtype(dtype):
    # Aligned X has beta-alpha=1/3 and fails margin=0.5. Orthogonal X passes.
    hidden = torch.tensor([[[0.0, 0.0, 3.0], [2.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0]]], dtype=dtype)
    before = hidden.clone()
    sentinel = object()
    stats = {}
    hook = evaluate.get_ablation_intervention(
        torch.tensor([1.0, 0.0, 0.0]), None, "vavg_aw_gate",
        margin=0.5, stats=stats,
    )
    output, rest = hook((hidden, sentinel), start_idx=1)
    assert rest is sentinel
    assert output.dtype == dtype
    assert torch.equal(hidden, before)
    assert torch.equal(output[:, :2], hidden[:, :2])
    assert not torch.equal(output[:, 2:], hidden[:, 2:])
    torch.testing.assert_close(output.float().norm(dim=-1), hidden.float().norm(dim=-1),
                               rtol=0.005, atol=0.005)
    assert stats["steered"] == 1
    assert stats["total"] == stats["weight_count"] == 2


@pytest.mark.parametrize("margin,steered", [(-2.0, 2), (2.0, 0)])
def test_parallel_and_antipodal_inputs_remain_finite(margin, steered):
    hidden = torch.tensor([[[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]])
    stats = {}
    output = evaluate.get_ablation_intervention(
        torch.tensor([1.0, 0.0, 0.0]), None, "vavg_aw_gate",
        margin=margin, stats=stats,
    )(hidden, start_idx=0)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.norm(dim=-1), hidden.norm(dim=-1))
    assert stats["total"] == 2
    assert stats["steered"] == steered
    if steered == 0:
        assert torch.equal(output, hidden)


def test_mean_control_is_exported_with_correct_component_flags(tmp_path):
    assert evaluate.VARIANTS.index("vavg_aw_gate") == evaluate.VARIANTS.index("vavg") + 1
    row = {"variant": "vavg_aw_gate", **evaluate.VARIANT_META["vavg_aw_gate"],
           "MC1": 0.4, "MC2": 0.6, "MC3": 0.5, "Avg": 0.5,
           "steer_token_ratio": 0.5,
           "mean_weights": {"alpha": 0.2, "beta": 0.8, "gamma": 0.0}}
    evaluate._write_outputs([row], tmp_path, {"fold_idx": 0})
    assert json.loads((tmp_path / "summary.json").read_text())["rows"] == [row]
    with (tmp_path / "table3_component_ablation.csv").open() as handle:
        exported = list(csv.DictReader(handle))[0]
    assert [exported[key] for key in ("MLP", "S2S", "AW", "Margin Gate")] == [
        "False", "False", "True", "True"
    ]
    assert (tmp_path / "table3_component_ablation.tex").exists()
