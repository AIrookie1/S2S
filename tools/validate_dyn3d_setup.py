#!/usr/bin/env python3
"""Static/project-layout checks for the reorganized Dyn3D + spherical tree."""
from __future__ import annotations
import py_compile
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = {"load_balance_ablation", "margin_ablation", "router_topk_ablation", "val_fraction_ablation", "weight_kappa_ablation"}
ANALYSES = {"steering_vector_ablation", "router_topk_position_analysis", "router_topk_frequency_analysis"}
GENERIC = {"boolq", "copa", "storycloze", "mmlu_global", "winogrande"}
MODELS = {"LLAMA", "QWEN"}


def main() -> int:
    errors: list[str] = []
    for p in ROOT.rglob("*.sh"):
        r = subprocess.run(["bash", "-n", str(p)], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            errors.append(f"bash -n {p.relative_to(ROOT)}: {r.stderr.strip()}")
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as exc:
            errors.append(f"py_compile {p.relative_to(ROOT)}: {exc}")

    for name in ABLATIONS:
        for f in ("README.md", "run.sh", "collect.py", "plot.py", "plot.sh"):
            if not (ROOT / "experiments" / name / f).is_file():
                errors.append(f"missing experiments/{name}/{f}")
        if not (ROOT / "hyperparams" / "dyn3d_sga" / "experiments" / f"{name}.sh").is_file():
            errors.append(f"missing hyperparams for {name}")
    for name in ANALYSES:
        if not (ROOT / "experiments" / name / "run.sh").is_file():
            errors.append(f"missing experiments/{name}/run.sh")
    for legacy in ("fig1a", "fig2a", "fig2b"):
        if (ROOT / "experiments" / legacy).exists():
            errors.append(f"legacy experiment directory remains: experiments/{legacy}")

    for method in ("dyn3d_sga", "spherical_steering"):
        for model in MODELS:
            base = ROOT / "hyperparams" / method / model
            for f in ("model.sh", "truthfulqa.sh"):
                if not (base / f).is_file():
                    errors.append(f"missing {base.relative_to(ROOT)}/{f}")
            for ds in GENERIC:
                if not (base / "generic" / f"{ds}.sh").is_file():
                    errors.append(f"missing {base.relative_to(ROOT)}/generic/{ds}.sh")

    # No generated result artifacts should live under experiments anymore.
    generated = {".png", ".pdf", ".svg", ".csv", ".json", ".jsonl"}
    for p in (ROOT / "experiments").rglob("*"):
        if p.is_file() and p.suffix.lower() in generated:
            errors.append(f"generated artifact still inside experiments: {p.relative_to(ROOT)}")

    # Catch references to deleted layouts/names in executable source. README/history text is allowed.
    stale = re.compile(r"experiments/model_config\.sh|hyperparams/(?:LLAMA|QWEN|generic)/|script/spherical/")
    for base in (ROOT / "experiments", ROOT / "script", ROOT / "method"):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {".sh", ".py"}:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if stale.search(text):
                    errors.append(f"stale layout reference in {p.relative_to(ROOT)}")

    # Verify Spherical-Steering core entrypoints and Dyn3D generic entrypoints exist.
    for f in ("steering.py", "get_activations.py", "get_prototypes.py", "evaluate_mc.py", "evaluate_llm_judge.py"):
        if not (ROOT / "method" / "spherical_steering" / f).is_file():
            errors.append(f"missing method/spherical_steering/{f}")
    for f in ("evaluate_generic.py", "get_activations_generic.py", "prompts_generic.py", "utils_generic.py"):
        if not (ROOT / "method" / "dyn3d_sga" / "generic" / f).is_file():
            errors.append(f"missing method/dyn3d_sga/generic/{f}")

    if errors:
        print("Validation failed:")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    print("Validation OK: Python/shell syntax, restored experiment files, reorganized hyperparameters, result hygiene, and method entrypoints all pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
