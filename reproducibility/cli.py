"""Unified, configuration-driven entry point for the paper and rebuttal experiments."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from reproducibility.config import ROOT, load_profile, load_registry
from reproducibility.run import Plan, build_plan


RUNTIME_ENVIRONMENT = (
    "MODEL MODEL_DIR MODEL_LABEL MODEL_KEY FAMILY MODELS MODEL_FAMILY LAYER FOLD_IDX FOLDS "
    "NUM_FOLDS VAL_FRACTION MLP_HIDDEN EPOCHS BATCH_SIZE LEARNING_RATE ALPHA BETA KAPPA "
    "STRENGTH SLERP_T WEIGHT_KAPPA MARGIN METHOD POSITIVE_MODE FEW_SHOT NO_INSTRUCTION "
    "NO_INTERVENTION EVAL_SPLIT EVAL_PRESET MAX_SAMPLES RUN_TRAIN RUN_EXTRACT RUN_EVAL "
    "RUN_PROTOTYPE DRY_RUN FEATURE_PATH OUTPUT_PATH OUTPUT_DIR EXPERIMENT_RESULT_DIR "
    "EXTRA_ARGS NOISE_LEVELS NOISE_SEED EXTRACT_BATCH_SIZE SEEDS HIDDEN_SIZES "
    "STRENGTH_VALUES SLERP_T_VALUES WEIGHT_KAPPA_VALUES MARGIN_VALUES LAYERS METHODS SEARCH_PROFILE "
    "LLAMA_LAYER QWEN_LAYER LLAMA_FOLD_IDX QWEN_FOLD_IDX LLAMA_MODEL QWEN_MODEL "
    "LLAMA_GPU QWEN_GPU NUM_HIDDEN_LAYERS LLAMA_NUM_HIDDEN_LAYERS QWEN_NUM_HIDDEN_LAYERS "
    "SUMMARY_DIR FIGURE_DIR OUTPUT_PREFIX"
).split()


def _format_arguments(arguments, context):
    result = []
    for argument in arguments:
        key = argument[1:-1] if argument.startswith("{") and argument.endswith("}") else None
        if key and isinstance(context.get(key), list):
            result.extend(str(value) for value in context[key])
        else:
            result.append(argument.format_map({k: str(v) for k, v in context.items()}))
    return result


def _relative_or_absolute(value):
    path = Path(value)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def make_plan(arguments):
    registry = load_registry()
    if arguments.experiment not in registry:
        raise ValueError("Unknown experiment; run `python -m reproducibility list`")
    spec = copy.deepcopy(registry[arguments.experiment])
    if getattr(arguments, "plot_only", False):
        if not spec.get("plot_entry"):
            raise ValueError("This experiment has no separate aggregate plotting stage")
        spec.update(id=spec["id"] + "-plot", title=spec["title"] + " (plot)",
                    backend="shell", entry=spec["plot_entry"], env=spec.get("plot_env", {}),
                    needs_model=False, required_fields=[], required_files=[], depends_on=[arguments.experiment])
    if spec["backend"] == "unavailable":
        raise ValueError(spec["note"])
    profile = load_profile(arguments.config, arguments.model)
    model = profile["model"]
    if model not in spec.get("models", ["llama", "qwen"]):
        raise ValueError("This published experiment supports: " + ", ".join(spec["models"]))
    runtime = profile["execution"]
    epochs = arguments.epochs or spec.get("default_epochs", profile["training"]["epochs"])
    folds = arguments.folds or spec.get("default_folds", [0, 1])
    if len(set(folds)) != len(folds):
        raise ValueError("Folds must be unique")
    if "fixed_folds" in spec and folds != spec["fixed_folds"]:
        raise ValueError("This diagnostic uses the fixed fold set " + str(spec["fixed_folds"]))
    work = Path(arguments.work_dir or runtime["work_dir"])
    if spec.get("fixed_work_dir") and _relative_or_absolute(work) != spec["fixed_work_dir"]:
        raise ValueError("This native supplementary entry expects --work-dir " + spec["fixed_work_dir"])
    model_dir = arguments.model_dir or profile["model_dir"]
    namespace = argparse.Namespace(
        suite=spec.get("suite", "native"), model=model, model_dir=model_dir,
        config=profile["config_path"], model_root=None,
        scaling_models=arguments.scaling_models or ["qwen05", "qwen15", "qwen3", "qwen14"],
        datasets=arguments.datasets or ["boolq", "copa", "storycloze", "mmlu_global", "winogrande"],
        folds=folds, work_dir=work, gpu=arguments.gpu or runtime["gpu"],
        device=runtime["device"], dtype=runtime["dtype"], epochs=epochs,
        bipo_epochs=20, seed=arguments.seed if arguments.seed is not None else spec.get("default_seed", runtime["seed"]),
        prompts=arguments.prompts, resume=arguments.resume,
    )
    if spec.get("fixed_seed_grid") and arguments.seed is not None:
        raise ValueError("This study retains its fixed seed grid " + str(spec["fixed_seed_grid"]))
    native_fixed = spec.get("native_execute") and spec["id"] != "r1-noise"
    if native_fixed and (epochs != 2000 or namespace.seed != 42):
        raise ValueError("This diagnostic retains its published seed/epoch protocol; use its native entry for a new protocol")
    plan = Plan(namespace)
    required_fields = spec.get("required_fields", [])
    context = {
        "python": sys.executable, "model": model, "model_name": profile["model_name"],
        "family": model.upper(), "model_dir": model_dir, "work_dir": str(work),
        "gpu": namespace.gpu, "epochs": epochs, "seed": namespace.seed,
        "folds": folds, "folds_text": " ".join(str(fold) for fold in folds),
        "feature": str(plan.feature), "feature_dir": str(work / "features"),
        "checkpoint_root": str(work / "checkpoints"), "prototype_dir": str(work / "prototypes"),
        "s2s_checkpoint0": str(plan.s2s(0)), "spherical_prototype0": str(plan.prototype(0)),
        "truth_judge_dir": arguments.truth_judge_dir or "<truth_judge_dir>",
        "info_judge_dir": arguments.info_judge_dir or "<info_judge_dir>",
        "prompts": arguments.prompts or "<prompts>",
        "generations": arguments.generations or "<generations>",
        "scores": arguments.scores or "<scores>",
        "judge_config": arguments.judge_config or "<judge_config>",
        "thresholds": arguments.thresholds, "safety_output": arguments.output_dir or str(work / "safety"),
        "search_profile": arguments.search_profile,
    }
    missing = [field for field in required_fields if getattr(arguments, field, None) is None]
    config_path = Path(profile["config_path"])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    llama = load_profile(model="llama")
    qwen = load_profile(model="qwen")
    environment = {
        "PYTHON_BIN": sys.executable, "PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": namespace.gpu, "GPU": namespace.gpu,
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "S2S_CONFIG": str(config_path),
        "LLAMA_MODEL_DIR": model_dir if model == "llama" else llama["model_dir"],
        "QWEN_MODEL_DIR": model_dir if model == "qwen" else qwen["model_dir"],
        "FEATURE_DIR": context["feature_dir"], "CHECKPOINT_ROOT": context["checkpoint_root"],
        "PROTOTYPE_DIR": context["prototype_dir"], "RESULT_ROOT": "results",
        "SEED": str(namespace.seed), "EPOCHS": str(epochs),
        "DEVICE": namespace.device, "DTYPE": namespace.dtype,
        "MODELS": model.upper(), "MODEL_KEY": model.upper(), "FAMILY": model.upper(),
        "REUSE_EXISTING": "1" if arguments.resume else "0",
    }
    if spec.get("dependency_profile") == "scaling":
        environment["DTYPE"] = "bfloat16"
    commands = []
    if spec["backend"] == "suite":
        command = [sys.executable, "-m", "reproducibility.run", spec["suite"],
                   "--config", profile["config_path"], "--model", model, "--model-dir", model_dir,
                   "--work-dir", str(work), "--gpu", namespace.gpu, "--seed", str(namespace.seed),
                   "--epochs", str(epochs), "--folds", *[str(fold) for fold in folds], "--execute"]
        if arguments.resume:
            command.append("--resume")
        if spec.get("accept_prompts"):
            command += ["--prompts", context["prompts"]]
        if spec.get("accept_datasets") and arguments.datasets:
            command += ["--datasets", *arguments.datasets]
        commands.append(dict(argv=command, env=environment))
    elif spec["backend"] == "python":
        command = [sys.executable, spec["entry"], *_format_arguments(spec.get("arguments", []), context)]
        if spec.get("accept_folds"):
            command += ["--folds", *[str(fold) for fold in folds]]
        options = set()
        for node in ast.walk(ast.parse((ROOT / spec["entry"]).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                options.update(value.value for value in node.args if isinstance(value, ast.Constant) and isinstance(value.value, str))
        if "--seed" in options and "--seed" not in command:
            command += ["--seed", str(namespace.seed)]
        elif namespace.seed != 42 and "--seed" not in options:
            raise ValueError("This native experiment has a fixed seed protocol")
        if arguments.epochs is not None and "--epochs" not in options:
            raise ValueError("This native entry has no --epochs override")
        if spec.get("native_execute"):
            command.append("--execute")
        if spec.get("native_resume") and arguments.resume:
            command.append("--resume")
        commands.append(dict(argv=command, env=environment))
    elif spec["backend"] == "shell":
        custom = {key: value.format_map({k: str(v) for k, v in context.items()}) for key, value in spec.get("env", {}).items()}
        fold_env = spec.get("fold_env")
        batches = folds if fold_env == "FOLD_IDX" else [None]
        for fold in batches:
            selected = dict(environment, **custom)
            if fold_env == "FOLD_IDX":
                selected[fold_env] = str(fold)
            elif fold_env:
                selected[fold_env] = context["folds_text"]
            commands.append(dict(argv=["bash", spec["entry"]], env=selected))
    elif spec["backend"] == "scaling_followup":
        if arguments.epochs is not None or arguments.seed is not None:
            raise ValueError("Follow-up sweeps inherit training/seed settings from their source checkpoints")
        for field in ("EPOCHS", "SEED", "DTYPE"):
            environment.pop(field, None)
        from experiments.rebuttal_R1W2_model_scaling.common import MODELS
        keys = arguments.scaling_models or spec["model_keys"]
        for key in keys:
            if key not in MODELS:
                raise ValueError("Unknown follow-up model: " + key)
            name = MODELS[key][0].split("/")[-1]
            commands.append(dict(argv=[sys.executable, spec["entry"], "--model", name], env=environment))
        commands.append(dict(argv=[sys.executable, spec["entry"], "--summarize"], env=environment))
    elif spec["backend"] == "scaling_batch":
        from experiments.rebuttal_R1W2_model_scaling.common import MODELS
        keys = arguments.scaling_models or spec["model_keys"]
        for key in keys:
            if key not in MODELS:
                raise ValueError("Unknown scaling model: " + key)
            name = MODELS[key][0].split("/")[-1]
            tag = "nf4" if key == "qwen14" else ("bfloat16_fullgpu_e250" if key in ("qwen35", "nemotron_nano8b") and epochs == 250 else "bfloat16")
            output = Path("results/experiment/rebuttal_R1W2_model_scaling") / name / tag
            command = [sys.executable, "experiments/rebuttal_R1W2_model_scaling/rebuttal_R1W2_run.py",
                       "--model", key, "--epochs", str(epochs), "--seed", str(namespace.seed), "--output-dir", str(output)]
            if key == "qwen14":
                command.append("--load-in-4bit")
            commands.append(dict(argv=command, env=environment))
    else:
        raise ValueError("Unsupported experiment backend")
    return {
        "schema_version": 1, "experiment": spec["id"], "title": spec["title"], "paper": spec["paper"],
        "dependency_profile": spec.get("dependency_profile", "main"),
        "config": profile["config_path"],
        "model": "model-batch" if spec["backend"] in ("scaling_batch", "scaling_followup") else model,
        "runtime": {"work_dir": str(work), "gpu": namespace.gpu, "seed": namespace.seed,
                    "epochs": None if spec["backend"] == "scaling_followup" else epochs, "folds": folds, "model_dir": model_dir},
        "requires": spec.get("depends_on", []), "native_seed_grid": spec.get("fixed_seed_grid", []), "missing_inputs": missing,
        "required_files": [context[field] for field in spec.get("required_files", [])],
        "needs_model": spec.get("needs_model", True), "note": spec.get("note"), "commands": commands,
    }


def execute_plan(plan):
    if plan["missing_inputs"]:
        flags = ["--" + value.replace("_", "-") for value in plan["missing_inputs"]]
        raise ValueError("Missing required inputs: " + ", ".join(flags))
    if plan["needs_model"] and not (ROOT / plan["runtime"]["model_dir"]).is_dir():
        raise FileNotFoundError("Download the model first, or set --model-dir to its local directory")
    missing = [name for name in plan["required_files"] if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite files; first run " + ", ".join(plan["requires"]))
    versions = {}
    for name in ("torch", "transformers", "numpy", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    work = ROOT / plan["runtime"]["work_dir"]
    receipt = work / "run_configs" / plan["experiment"] / (plan["model"] + ".json")
    signature_plan = copy.deepcopy(plan)
    for item in signature_plan["commands"]:
        item["argv"] = [argument for argument in item["argv"] if argument != "--resume"]
        item["env"].pop("REUSE_EXISTING", None)
    from reproducibility.run import source_digest
    signature = hashlib.sha256(json.dumps({"plan": signature_plan, "versions": versions,
        "source_sha256": source_digest()}, sort_keys=True).encode()).hexdigest()
    resolved = {"plan": plan, "versions": versions, "execution_signature": signature}
    if receipt.exists() and json.loads(receipt.read_text()).get("execution_signature") != signature:
        raise ValueError("Resolved configuration changed; choose a fresh --work-dir")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(resolved, indent=2) + "\n")
    for item in plan["commands"]:
        environment = dict(os.environ)
        for key in RUNTIME_ENVIRONMENT:
            environment.pop(key, None)
        environment.update(item["env"])
        print("$ " + shlex.join(item["argv"]), flush=True)
        subprocess.run(item["argv"], cwd=ROOT, env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="List registered paper/rebuttal experiments")
    listing.add_argument("--group", choices=["main", "rebuttal", "supplement", "external"])
    listing.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="Preview or execute one registered experiment")
    run.add_argument("experiment")
    run.add_argument("--config", type=Path)
    run.add_argument("--model", choices=["llama", "qwen"])
    run.add_argument("--model-dir")
    run.add_argument("--gpu")
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--folds", type=int, nargs="+", choices=[0, 1])
    run.add_argument("--work-dir")
    run.add_argument("--datasets", nargs="+", choices=["boolq", "copa", "storycloze", "mmlu_global", "winogrande"])
    run.add_argument("--scaling-models", nargs="+")
    run.add_argument("--truth-judge-dir")
    run.add_argument("--info-judge-dir")
    run.add_argument("--prompts")
    run.add_argument("--generations")
    run.add_argument("--scores")
    run.add_argument("--judge-config")
    run.add_argument("--thresholds", type=float, nargs="+", default=[.5, .7, .9])
    run.add_argument("--output-dir")
    run.add_argument("--search-profile", choices=["fast", "balanced", "full"], default="fast")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    plotting = sub.add_parser("plot", help="Render an aggregate figure after its model runs finish")
    plotting.add_argument("experiment")
    plotting.add_argument("--config", type=Path)
    plotting.add_argument("--model", choices=["llama", "qwen"])
    plotting.add_argument("--work-dir")
    plotting.add_argument("--execute", action="store_true")
    plotting.add_argument("--dry-run", action="store_true")
    plotting.set_defaults(plot_only=True, model_dir=None, gpu=None, seed=None, epochs=None,
        folds=None, datasets=None, scaling_models=None, truth_judge_dir=None, info_judge_dir=None,
        prompts=None, generations=None, scores=None, judge_config=None, thresholds=[.5,.7,.9],
        output_dir=None, search_profile="fast", resume=False)
    export = sub.add_parser("export", help="Create a clean, scanned copy without Git history")
    export.add_argument("--output", type=Path, default=Path("dist/s2s-anonymous"))
    verify = sub.add_parser("verify", help="Verify source hashes in a checkout/export")
    verify.add_argument("directory", nargs="?", type=Path, default=ROOT)
    verify.add_argument("--allow-git", action="store_true")
    verify.add_argument("--strict", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "list":
            entries = [entry for entry in load_registry().values() if not arguments.group or entry["group"] == arguments.group]
            if arguments.json:
                print(json.dumps(entries, indent=2))
            else:
                for entry in entries:
                    suffix = " [external inputs not bundled]" if entry["backend"] == "unavailable" else ""
                    print(f"{entry['id']:<24} {entry['paper']:<29} {entry['title']}{suffix}")
        elif arguments.action in ("run", "plot"):
            if arguments.epochs is not None and arguments.epochs <= 0:
                raise ValueError("--epochs must be positive")
            plan = make_plan(arguments)
            if arguments.execute:
                execute_plan(plan)
            else:
                print(json.dumps({"dry_run": True, **plan}, indent=2))
        elif arguments.action == "export":
            from reproducibility.build_release import build
            print(json.dumps(build(ROOT, arguments.output, include_results=True), indent=2))
        elif arguments.action == "verify":
            from reproducibility.verify_release import verify as check
            findings = check(arguments.directory, arguments.allow_git, arguments.strict)
            print(json.dumps({"passed": not findings, "findings": findings}, indent=2))
            if findings:
                raise SystemExit(1)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
