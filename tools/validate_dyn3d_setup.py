#!/usr/bin/env python3
"""Static/project-layout checks for the reorganized Dyn3D + spherical tree."""
from __future__ import annotations
import py_compile
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = {"margin_ablation", "val_fraction_ablation", "weight_kappa_ablation"}
ANALYSES = {"steering_vector_ablation"}
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

    # steering_vector_ablation must use one base pairs.npz per layer and derive
    # folds in memory; the old helper that materialized fold NPZ copies is gone.
    steering_dir = ROOT / "experiments" / "steering_vector_ablation"
    if (steering_dir / "prepare_fold_features.py").exists():
        errors.append(
            "steering_vector_ablation still contains prepare_fold_features.py"
        )
    for name in ("run.sh", "evaluate.py"):
        p = steering_dir / name
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "_fold{fold}" in text or "prepare_fold_features.py" in text:
                errors.append(
                    f"steering_vector_ablation/{name} still depends on materialized fold features"
                )

    for method in ("dyn3d_sga", "spherical_steering"):
        for model in MODELS:
            base = ROOT / "hyperparams" / method / model
            for f in ("model.sh", "truthfulqa.sh"):
                if not (base / f).is_file():
                    errors.append(f"missing {base.relative_to(ROOT)}/{f}")
            for ds in GENERIC:
                if not (base / "generic" / f"{ds}.sh").is_file():
                    errors.append(f"missing {base.relative_to(ROOT)}/generic/{ds}.sh")

    # The two L2S variants are TruthfulQA-only and must stay independently named.
    for method in ("l2s_linear", "l2s_slerp"):
        for model in MODELS:
            base = ROOT / "hyperparams" / method / model
            for f in ("model.sh", "truthfulqa.sh"):
                if not (base / f).is_file():
                    errors.append(f"missing {base.relative_to(ROOT)}/{f}")
            script_base = ROOT / "script" / method / model
            for f in ("run.sh", "run_train.sh", "run_eval.sh", "run_llm_judge.sh"):
                if not (script_base / f).is_file():
                    errors.append(f"missing {script_base.relative_to(ROOT)}/{f}")
        method_base = ROOT / "method" / method
        for f in ("model.py", "steering.py", "train_l2s.py", "evaluate_mc.py", "evaluate_llm_judge.py"):
            if not (method_base / f).is_file():
                errors.append(f"missing {method_base.relative_to(ROOT)}/{f}")
    for legacy in (ROOT / "method" / "l2s", ROOT / "script" / "l2s", ROOT / "hyperparams" / "l2s"):
        if legacy.exists():
            errors.append(f"ambiguous legacy L2S directory remains: {legacy.relative_to(ROOT)}")

    # No generated result artifacts should live under experiments anymore.
    generated = {".png", ".pdf", ".svg", ".csv", ".json", ".jsonl"}
    for p in (ROOT / "experiments").rglob("*"):
        if p.is_file() and p.suffix.lower() in generated:
            errors.append(f"generated artifact still inside experiments: {p.relative_to(ROOT)}")

    # Catch literal path-template placeholders inside bash ${VAR:-default} expansions.
    # bash -n accepts these, but a literal closing brace such as {layer} can terminate
    # the outer parameter expansion early at runtime.
    literal_template = re.compile(r"(?<!\$)\{(?:layer|fold)\}")
    for p in ROOT.rglob("*.sh"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if ":-" in line and literal_template.search(line):
                errors.append(
                    f"unsafe literal template in bash default expansion: "
                    f"{p.relative_to(ROOT)}:{lineno}: {line.strip()}"
                )

    # Catch references to deleted layouts/names in executable source. README/history text is allowed.
    stale = re.compile(r"experiments/model_config\.sh|hyperparams/(?:LLAMA|QWEN|generic)/|script/spherical/")
    for base in (ROOT / "experiments", ROOT / "script", ROOT / "method"):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {".sh", ".py"}:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if stale.search(text):
                    errors.append(f"stale layout reference in {p.relative_to(ROOT)}")

    # Verify the direct-MLP Dyn3D core has fully replaced the old basis-routing core.
    for f in ("dyn_mlp.py", "train_dyn_mlp.py", "evaluate_dyn_mlp.py", "evaluate_3d_dyn.py"):
        if not (ROOT / "method" / "dyn3d_sga" / f).is_file():
            errors.append(f"missing method/dyn3d_sga/{f}")
    for obsolete in ("dynSVD.py", "train_dynSVD.py", "evaluate_dynSVD.py"):
        if (ROOT / "method" / "dyn3d_sga" / obsolete).exists():
            errors.append(f"obsolete Dyn3D file remains: method/dyn3d_sga/{obsolete}")
    for obsolete_dir in (
        "fig2",
        "load_balance_ablation",
        "router_topk_ablation",
        "router_topk_position_analysis",
        "router_topk_frequency_analysis",
    ):
        if (ROOT / "experiments" / obsolete_dir).exists():
            errors.append(f"obsolete basis-routing experiment remains: experiments/{obsolete_dir}")

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
    print("Validation OK: Python/shell syntax, direct-MLP Dyn3D core, split L2S-LINEAR/L2S-SLERP layout, experiment layout, hyperparameters, result hygiene, and method entrypoints all pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
