#!/usr/bin/env python3
"""Static and lightweight numerical checks for the Dyn2D addition."""
from __future__ import annotations

import math
import py_compile
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn2d.steering import get_2d_dynamic_intervention, slerp
from method.dyn3d_sga.dyn_mlp import DirectVectorMLP


def main() -> int:
    errors: list[str] = []

    required = [
        "method/dyn2d/steering.py",
        "method/dyn2d/evaluate_2d_dyn.py",
        "method/dyn2d/evaluate_llm_judge.py",
        "method/dyn2d/evaluate_llm_judge_2d.py",
        "method/dyn2d/generic/evaluate_generic.py",
        "experiments/dyn2d_ablation/run.sh",
        "experiments/dyn2d_ablation/evaluate_grid.py",
        "experiments/dyn2d_ablation/plot.py",
        "hyperparams/dyn2d/experiments/dyn2d_ablation.sh",
    ]
    for rel in required:
        if not (ROOT / rel).is_file():
            errors.append(f"missing {rel}")

    for model in ("LLAMA", "QWEN"):
        for rel in (
            f"hyperparams/dyn2d/{model}/model.sh",
            f"hyperparams/dyn2d/{model}/truthfulqa.sh",
            f"script/dyn2d/{model}/run_2d_extract.sh",
            f"script/dyn2d/{model}/run_2d_train.sh",
            f"script/dyn2d/{model}/run_2d_eval.sh",
            f"script/dyn2d/{model}/run_2d_dyn.sh",
            f"script/dyn2d/{model}/run_llm_judge.sh",
            f"script/dyn2d/{model}/run_llm_judge_2d.sh",
        ):
            if not (ROOT / rel).is_file():
                errors.append(f"missing {rel}")
        for dataset in ("boolq", "copa", "storycloze", "mmlu_global", "winogrande"):
            for rel in (
                f"hyperparams/dyn2d/{model}/generic/{dataset}.sh",
                f"script/dyn2d/{model}/generic/{dataset}.sh",
            ):
                if not (ROOT / rel).is_file():
                    errors.append(f"missing {rel}")

    for base in (ROOT / "method" / "dyn2d", ROOT / "experiments" / "dyn2d_ablation"):
        for path in base.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:
                errors.append(f"py_compile {path.relative_to(ROOT)}: {exc}")

    for base in (ROOT / "script" / "dyn2d", ROOT / "hyperparams" / "dyn2d", ROOT / "experiments" / "dyn2d_ablation"):
        for path in base.rglob("*.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            if proc.returncode:
                errors.append(f"bash -n {path.relative_to(ROOT)}: {proc.stderr.strip()}")

    try:
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0]])
        if not torch.allclose(slerp(x, y, 0.0), x, atol=1e-7):
            errors.append("SLERP t=0 endpoint mismatch")
        if not torch.allclose(slerp(x, y, 1.0), y, atol=1e-7):
            errors.append("SLERP t=1 endpoint mismatch")
        expected_mid = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)]])
        if not torch.allclose(slerp(x, y, 0.5), expected_mid, atol=1e-6):
            errors.append("SLERP orthogonal midpoint mismatch")
        anti_mid = slerp(x, -x, 0.5)
        if not torch.isfinite(anti_mid).all():
            errors.append("SLERP antipodal midpoint is non-finite")
        if not torch.allclose(
            torch.linalg.vector_norm(anti_mid, dim=-1), torch.ones(1), atol=1e-6
        ):
            errors.append("SLERP antipodal midpoint is not unit norm")

        predictor = DirectVectorMLP(2, 3, 2)
        with torch.no_grad():
            for parameter in predictor.parameters():
                parameter.zero_()
            predictor.fc2.bias[:] = torch.tensor([1.0, 1.0])
        avg = torch.tensor([1.0, 0.0])
        hook = get_2d_dynamic_intervention(
            avg,
            predictor,
            strength=2.0,
            slerp_t=0.5,
        )
        hidden = torch.tensor([[[5.0, 5.0], [10.0, 20.0], [30.0, 40.0]]])
        output = hook(hidden, start_idx=1)
        expected = hidden.clone()
        expected[:, 1:, :] += 2.0
        if not torch.allclose(output, expected, atol=1e-5):
            errors.append("Dyn2D hook does not match h + strength * D_scaled")
    except Exception as exc:
        errors.append(f"numerical check failed: {exc}")

    if errors:
        print("Dyn2D validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Dyn2D validation OK: layout, Python/shell syntax, SLERP endpoints, "
        "antipodal stability, and additive hook formula all pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
