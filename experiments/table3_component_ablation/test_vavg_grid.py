"""Check ranking, checkpoint reuse, interruption recovery and grid validation."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.table3_component_ablation.vavg_grid import run_grid, validate_grid


def result(mc1, mc2=0.6, mc3=0.5):
    return ({"variant": "vavg_aw_gate", "MC1": mc1, "MC2": mc2, "MC3": mc3,
             "Avg": (mc1 + mc2 + mc3) / 3, "steer_token_ratio": 0.5}, {"metrics": {"MC1": mc1}})


def test_ranks_by_mc1_and_can_rerank_cached_cells_by_avg(tmp_path):
    calls = []

    def evaluate(kappa, margin):
        calls.append((kappa, margin))
        return result(0.8, 0.3, 0.3) if kappa == 1 else result(0.7, 0.9, 0.9)

    kwargs = dict(kappas=[1.0, 2.0], margins=[0.5], output_dir=tmp_path,
                  context={"eval_split": "test"}, evaluate=evaluate, save_details=True)
    best = run_grid(**kwargs)
    assert best["weight_kappa"] == 1
    assert calls == [(1, 0.5), (2, 0.5)]
    calls.clear()
    best = run_grid(**kwargs, resume=True, selection_metric="Avg")
    assert best["weight_kappa"] == 2
    assert not calls
    report = json.loads((tmp_path / "best_config.json").read_text())
    assert report["metadata"]["complete"]
    assert report["metadata"]["selection_metric"] == "Avg"
    assert len((tmp_path / "grid_summary.csv").read_text().splitlines()) == 3
    assert len(list((tmp_path / "details").glob("*.json"))) == 2


def test_interrupted_grid_preserves_progress_and_recovers_corrupt_cell(tmp_path):
    def interrupt(kappa, margin):
        if kappa == 2:
            raise RuntimeError("simulated interruption")
        return result(0.4)

    kwargs = dict(kappas=[1.0, 2.0], margins=[0.5], output_dir=tmp_path,
                  context={"eval_split": "test"})
    with pytest.raises(RuntimeError, match="interruption"):
        run_grid(**kwargs, evaluate=interrupt)
    report = json.loads((tmp_path / "best_config.json").read_text())
    assert report["metadata"]["completed"] == 1
    assert not report["metadata"]["complete"]
    calls = []

    def evaluate(kappa, margin):
        calls.append(kappa)
        return result(0.5)

    run_grid(**kwargs, evaluate=evaluate, resume=True)
    assert calls == [2]
    calls.clear()
    first = next((tmp_path / "cells").glob("kappa1.0*.json"))
    first.write_text("broken JSON")
    run_grid(**kwargs, evaluate=evaluate, resume=True)
    assert calls == [1]


def test_resume_refuses_changed_context(tmp_path):
    kwargs = dict(kappas=[1.0], margins=[0.5], output_dir=tmp_path,
                  evaluate=lambda *args: result(0.5))
    run_grid(**kwargs, context={"eval_split": "test", "seed": 1})
    with pytest.raises(ValueError, match="different grid/context"):
        run_grid(**kwargs, context={"eval_split": "test", "seed": 2}, resume=True)


@pytest.mark.parametrize("kappas,margins", [([], [0.5]), ([1], []), ([0], [0.5]),
                                          ([-1], [0.5]), ([math.nan], [0.5]),
                                          ([1], [math.inf])])
def test_invalid_grids_are_rejected(kappas, margins):
    with pytest.raises(ValueError):
        validate_grid(kappas, margins)


def test_grid_deduplicates_and_rejects_invalid_metrics(tmp_path):
    assert validate_grid([1, 1, 2], [0.5, 0.5]) == ([1, 2], [0.5])
    with pytest.raises(ValueError, match="invalid grid result"):
        run_grid(kappas=[1], margins=[0.5], output_dir=tmp_path,
                 context={"eval_split": "test"}, evaluate=lambda *args: result(math.nan))


def test_primary_score_ties_use_avg_then_smaller_parameters(tmp_path):
    best = run_grid(kappas=[2.0, 1.0], margins=[0.7, 0.5], output_dir=tmp_path,
                    context={"eval_split": "test"}, evaluate=lambda *args: result(0.5))
    assert (best["weight_kappa"], best["margin"]) == (1, 0.5)
