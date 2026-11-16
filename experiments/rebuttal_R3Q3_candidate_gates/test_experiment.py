"""Boundary, native-hook fidelity and question-level weighting checks."""
import numpy as np
import pytest
import torch

from experiments.rebuttal_R3Q3_candidate_gates.analyze import paired_bootstrap
from experiments.rebuttal_R3Q3_candidate_gates.evaluate import GateRecorder, window_counts
from method.S2S_R.predictor import DirectVectorMLP
from method.S2S_R.evaluate_S2S_R import get_s2s_r_intervention


def test_answer_and_score_windows_have_opposite_endpoints():
    s = window_counts(torch.tensor([True, False, False]))
    assert s["answer_tokens"] == {"tokens": 2, "triggered": 0, "rate": 0.0}
    assert s["scored_answer_predictors"] == {"tokens": 2, "triggered": 1, "rate": 0.5}
    assert s["legacy_intervention_segment"]["tokens"] == 3
    with pytest.raises(ValueError):
        window_counts(torch.tensor([True]))


@pytest.mark.parametrize("as_tuple", [False, True])
def test_recorder_preserves_native_hook_and_observe_returns_original(as_tuple):
    torch.manual_seed(3)
    avg, hidden = torch.randn(12), torch.randn(1, 6, 12)
    predictor = DirectVectorMLP(12, 8, 12).eval()
    output = (hidden, "untouched") if as_tuple else hidden
    with torch.no_grad():
        native = get_s2s_r_intervention(avg, predictor, weight_kappa=3, margin=-0.7)(output, start_idx=2)
        apply = GateRecorder(avg, predictor, 3, -0.7, "apply")
        observe = GateRecorder(avg, predictor, 3, -0.7, "observe")
        applied, observed = apply(output, start_idx=2), observe(output, start_idx=2)
    assert observed is output
    torch.testing.assert_close(applied[0] if as_tuple else applied, native[0] if as_tuple else native)
    assert apply.records == observe.records
    assert apply.records[0]["windows"]["answer_tokens"]["tokens"] == 3
    if as_tuple:
        assert applied[1] == "untouched"


def test_question_paired_difference_and_stratification():
    s = paired_bootstrap([0.0, 0.8], [1.0, 0.8], [0, 1], 1000, 42)
    assert s["incorrect_minus_correct"] == pytest.approx(0.5)
    assert s["difference_ci95"] == pytest.approx([0.5, 0.5])
    assert s["questions_incorrect_higher"] == 1
    assert s["questions_tied"] == 1
    zero = paired_bootstrap([0.2, 0.8], [0.2, 0.8], [0, 0], 1000, 42)
    assert zero["difference_ci95"] == [0, 0]


def test_baukit_dispatches_recording_closure_with_start_boundary():
    from functools import partial
    from baukit import TraceDict
    torch.manual_seed(4)
    block = torch.nn.Sequential(torch.nn.Identity())
    recorder = GateRecorder(torch.randn(12), DirectVectorMLP(12, 8, 12).eval(), 3, -0.7, "observe")
    hidden = torch.randn(1, 6, 12)
    with torch.no_grad(), TraceDict(block, ["0"], edit_output=partial(recorder.as_hook(), start_idx=2)):
        observed = block(hidden)
    torch.testing.assert_close(observed, hidden)
    assert len(recorder.records) == 1
    assert recorder.records[0]["windows"]["answer_tokens"]["tokens"] == 3
