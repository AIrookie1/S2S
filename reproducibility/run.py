"""Print or execute explicit, isolated reproduction plans (stdlib-only planner)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import importlib.metadata
import platform
import re
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama": ("llama3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", 14),
    "qwen": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 19),
}
SCALING = ("qwen05", "qwen15", "qwen3", "qwen14")
MARGINS = (-0.8, -0.7, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    paths = [ROOT / "test.py", *sorted((ROOT / "method").rglob("*.py")),
             *sorted((ROOT / "truthfulqa").rglob("*.py")),
             *sorted((ROOT / "reproducibility").rglob("*.py")),
             *sorted((ROOT / "experiments").rglob("*.py")),
             *sorted((ROOT / "hyperparams").rglob("*.sh")),
             ROOT / "TruthfulQA/data/v1/TruthfulQA.csv"]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): digest(p)
                                      for p in paths if p.is_file()}, sort_keys=True).encode()).hexdigest()


class Plan:
    def __init__(self, args):
        self.args = args
        self.name, default_dir, self.layer = MODELS[args.model]
        self.model_dir = args.model_dir or default_dir
        self.work = args.work_dir
        self.feature = self.work / "features" / f"{self.name}_layer{self.layer}.npz"
        self.dyn_dir = self.work / "checkpoints/dyn3d_sga/truthfulqa" / self.name
        self.l2s_dir = self.work / "checkpoints/l2s_linear/truthfulqa" / self.name
        self.additive_dir = self.work / "additive_vectors" / self.name
        self.proto_dir = self.work / "prototypes" / self.name
        self.jobs = []
        self.seen = set()

    def add(self, name, command, outputs=(), env=None):
        if name not in self.seen:
            self.seen.add(name)
            self.jobs.append(dict(name=name, command=[str(x) for x in command],
                                  outputs=[str(x) for x in outputs], env=env or {}))

    def py(self, path, *args):
        return [sys.executable, path, *args]

    def entry(self, method, script, *args):
        return self.py("test.py", "--method-dir", method, "--entry-file", script, "--", *args)

    def dyn(self, fold):
        return self.dyn_dir / f"{self.name}_layer{self.layer}_dynMLP_fold{fold}_mean.npz"

    def l2s(self, fold):
        return self.l2s_dir / f"{self.name}_layer{self.layer}_l2s_linear_fold{fold}.npz"

    def prototype(self, fold):
        # Native spherical fold IDs refer to the TEST half; S2S IDs to TRAIN.
        return self.proto_dir / f"{self.feature.stem}_fold{1-fold}.npz"

    def prepare(self, l2s=False, prototypes=False, train=True):
        a = self.args
        self.add(f"extract-{self.name}", self.py("method/dyn3d_sga/extract_features.py", self.name,
                 "--model_dir", self.model_dir, "--layer", self.layer, "--save_dir", self.feature.parent,
                 "--output_name", self.feature.name, "--device", a.device, "--dtype", a.dtype,
                 "--split_seed", a.seed), [self.feature])
        for f in (a.folds if train else []):
            shared = ["--feature_path", self.feature, "--num_folds", 2, "--fold_idx", f,
                      "--val_fraction", 0.5, "--mlp_hidden", 256, "--epochs", a.epochs,
                      "--batch_size", 256, "--learning_rate", 0.001, "--seed", a.seed,
                      "--device", a.device]
            self.add(f"train-s2s-{self.name}-{f}", self.py("method/dyn3d_sga/train_dyn_mlp.py",
                     *shared, "--positive_mode", "mean", "--output_path", self.dyn(f)), [self.dyn(f)])
            if l2s:
                self.add(f"train-l2s-{self.name}-{f}", self.py("method/l2s_linear/train_l2s.py",
                         *shared, "--output_path", self.l2s(f)), [self.l2s(f)])
        if prototypes:
            outputs = [self.proto_dir / f"{self.feature.stem}_fold{f}.npz" for f in (0, 1)]
            self.add(f"prototypes-{self.name}", self.py("method/spherical_steering/get_prototypes.py",
                     "--feature_file", self.feature, "--num_folds", 2, "--save_dir", self.proto_dir), outputs)

            self.add(f"additive-vectors-{self.name}", self.py("-m", "reproducibility.prepare_additive",
                     "--features", self.feature, "--output-dir", self.additive_dir),
                     [self.additive_dir / f"fold{f}.npz" for f in (0, 1)])

    def common(self, fold):
        return [self.name, "--model_dir", self.model_dir, "--layer", self.layer,
                "--device", self.args.device, "--seed", self.args.seed, "--preset", "qa"]

    def main(self):
        self.prepare(l2s=True, prototypes=True)
        for f in self.args.folds:
            out = self.work / "main" / self.name / f"fold{f}"
            common = self.common(f)
            for method in ("baseline", "additive", "l2s", "spherical", "s2s_l", "s2s_r"):
                target = out / f"{method}.json"
                if method in ("baseline", "additive"):
                    extra = ["--no_intervention"] if method == "baseline" else [
                        "--vector_source", "v", "--no_normalize_v", "--alpha", 1.0]
                    cmd = self.entry("dyn3d_sga", "eval_utils.py", *common,
                        "--dtype", self.args.dtype, "--prototype_path", self.additive_dir / f"fold{f}.npz",
                        "--eval_split", "test", *extra, "--output_path", target, "--save_details")
                elif method == "spherical":
                    alpha, beta = (0.7, -0.15) if self.args.model == "llama" else (0.6, 0.4)
                    cmd = self.entry("spherical_steering", "evaluate_mc.py", *common,
                        "--prototype_path", self.prototype(f), "--kappa", 20, "--alpha", alpha,
                        "--beta", beta, "--output_path", target, "--save_details")
                elif method == "l2s":
                    cmd = self.entry("l2s_linear", "evaluate_mc.py", *common, "--dtype", self.args.dtype,
                        "--checkpoint_path", self.l2s(f), "--eval_split", "test", "--alpha", 1.0,
                        "--output_path", target, "--save_details")
                elif method == "s2s_l":
                    cmd = self.entry("dyn2d", "evaluate_2d_dyn.py", *common, "--dtype", self.args.dtype,
                        "--checkpoint_path", self.dyn(f), "--method", "S2S_L_Static", "--strength", 3.0,
                        "--slerp_t", 0.2, "--eval_split", "test", "--output_path", target, "--save_details")
                else:
                    cmd = self.entry("dyn3d_sga", "evaluate_3d_dyn.py", *common, "--dtype", self.args.dtype,
                        "--checkpoint_path", self.dyn(f), "--weight_kappa", 3, "--margin", -0.7,
                        "--eval_split", "test", "--output_path", target, "--save_details")
                self.add(f"main-{self.name}-{f}-{method}", cmd, [target])

        if set(self.args.folds) == {0, 1}:
            report = self.work / "reports" / self.name
            self.add(f"collect-main-{self.name}", self.py("-m", "reproducibility.collect",
                "--work-dir", self.work, "--output", report, "--models", self.name),
                [report / "main_per_fold.csv", report / "main_averages.csv"])

    def component(self):
        self.prepare()
        for f in self.args.folds:
            out = self.work / "component" / self.name / f"fold{f}"
            self.add(f"component-{self.name}-{f}", self.py("experiments/component_ablation/evaluate.py",
                *self.common(f), "--dtype", self.args.dtype, "--checkpoint_path", self.dyn(f),
                "--expected_fold", f, "--weight_kappa", 3, "--margin", -0.7,
                "--variants", "vavg", "vavg_aw_gate", "mlp", "s2s_fixed", "s2s_adaptive", "s2s_full",
                "--eval_split", "test", "--output_dir", out, "--save_details"), [out / "summary.json"])
            review_out = self.work / "component_review_control" / self.name / f"fold{f}"
            self.add(f"review-control-{self.name}-{f}", self.py("experiments/component_ablation/evaluate.py",
                *self.common(f), "--dtype", self.args.dtype, "--checkpoint_path", self.dyn(f),
                "--expected_fold", f, "--weight_kappa", 2.5, "--margin", 0.965,
                "--variants", "vavg_aw_gate", "--eval_split", "test", "--output_dir", review_out,
                "--save_details"), [review_out / "summary.json"])

    def generic(self):
        family = self.args.model.upper()
        for dataset in self.args.datasets:
            for directory, method in (("dyn3d", "S2S_R"), ("dyn2d", "S2S_L_Static"),
                                       ("l2s_linear", "l2s_linear"), ("spherical_steering", "spherical_steering")):
                env = dict(PYTHON_BIN=sys.executable, LLAMA_MODEL_DIR=self.model_dir, QWEN_MODEL_DIR=self.model_dir,
                    GPU=self.args.gpu, DEVICE=self.args.device, DTYPE=self.args.dtype, SEED=str(self.args.seed),
                    EPOCHS=str(self.args.epochs), RUN_EXTRACT="1", RUN_TRAIN="1", RUN_EVAL="1", RUN_PROTOTYPE="1",
                    FEATURE_DIR=str(self.work / "features"), CHECKPOINT_ROOT=str(self.work / "checkpoints"),
                    PROTOTYPE_DIR=str(self.work / "prototypes"), RESULT_ROOT=str(self.work),
                    GEN_FEATURE_DIR=str(self.work / "generic/features" / directory / self.name),
                    GEN_CHECKPOINT_DIR=str(self.work / "generic/checkpoints" / directory / self.name),
                    GEN_PROTO_DIR=str(self.work / "generic/prototypes" / directory / self.name),
                    RESULT_DIR=str(self.work / "generic" / dataset / method / self.name))
                if directory in ("dyn3d", "dyn2d"):
                    env["METHOD"] = method
                    filename = f"{self.name}_{dataset}_{method}_mc1.json"
                elif directory == "l2s_linear":
                    env["METHOD"] = "L2S_LINEAR"
                    filename = f"{self.name}_{dataset}_l2s_linear_mc1.json"
                else:
                    filename = f"{self.name}_{dataset}_mc1.json"
                self.add(f"generic-{self.name}-{dataset}-{method}",
                    ["bash", f"script/{directory}/{family}/generic/run_dataset.sh", dataset],
                    [Path(env["RESULT_DIR"]) / filename], env)
                if directory == "dyn3d":
                    cfg = ROOT / "hyperparams/dyn3d_sga" / family / "generic" / (dataset + ".sh")
                    match = re.search(r'LAYER="\$\{LAYER:-(\d+)\}"', cfg.read_text())
                    if not match:
                        raise ValueError("Cannot determine generic layer from " + str(cfg))
                    layer = int(match.group(1))
                    ckpt = Path(env["GEN_CHECKPOINT_DIR"]) / f"{self.name}_{dataset}_l{layer}_dyn3d.npz"
                    target = self.work / "generic" / dataset / "baseline" / self.name / "baseline.json"
                    self.add(f"generic-{self.name}-{dataset}-baseline", self.py(
                        "method/dyn3d_sga/generic/evaluate_generic.py", "--model_name", self.name,
                        "--model_dir", self.model_dir, "--dataset", dataset, "--layer", layer,
                        "--checkpoint_path", ckpt, "--disable_steering", "--dtype", self.args.dtype,
                        "--seed", self.args.seed, "--output_path", target), [target])

    def gate(self):
        self.prepare()
        out = self.work / "gate" / self.name
        self.add(f"gate-{self.name}", self.py("experiments/gate_correctness/evaluate.py",
            "--model_name", self.name, "--model_dir", self.model_dir, "--checkpoint_dir", self.dyn_dir,
            "--layer", self.layer, "--folds", *self.args.folds, "--margin", -0.7,
            "--weight_kappa", 3, "--dtype", self.args.dtype, "--seed", self.args.seed,
            "--output_dir", out), [out / "summary.json"])

    def margin(self):
        self.prepare()
        eval_root = self.work / "margin/evaluations"
        for f in self.args.folds:
            for margin in MARGINS:
                tag = str(margin).replace("-", "m").replace(".", "p")
                out = eval_root / self.name / f"fold{f}" / f"margin{tag}_eval.json"
                self.add(f"margin-{self.name}-{f}-{tag}", self.entry("dyn3d_sga", "evaluate_3d_dyn.py",
                    *self.common(f), "--dtype", self.args.dtype, "--checkpoint_path", self.dyn(f),
                    "--weight_kappa", 3, "--margin", margin, "--output_path", out, "--save_details"), [out])
        out = self.work / "margin_gate" / self.name
        self.add(f"margin-gate-{self.name}", self.py("experiments/margin_gate_correctness/evaluate.py",
            "--model_name", self.name, "--model_dir", self.model_dir, "--checkpoint_dir", self.dyn_dir,
            "--layer", self.layer, "--folds", *self.args.folds, "--margins", *MARGINS,
            "--eval_root", eval_root, "--dtype", self.args.dtype, "--seed", self.args.seed,
            "--output_dir", out), [out / "summary.json"])

    def noise(self):
        self.prepare(l2s=True)
        for f in self.args.folds:
            out = self.work / "noise" / self.name / f"fold{f}"
            env = dict(FAMILY=self.args.model.upper(), MODEL_DIR_OVERRIDE=self.model_dir,
                LLAMA_MODEL_DIR=self.model_dir, QWEN_MODEL_DIR=self.model_dir,
                FOLD_IDX=str(f), FOLD_IDX_OVERRIDE=str(f), FEATURE_PATH=str(self.feature),
                DYN_FEATURE_PATH=str(self.feature), RESULT_DIR=str(out), PYTHON_BIN=sys.executable,
                GPU=self.args.gpu, DEVICE=self.args.device, DTYPE=self.args.dtype, SEED=str(self.args.seed),
                RUN_TRAIN="auto", NOISE_SEED=str(self.args.seed),
                EPOCHS=str(self.args.epochs), NOISE_LEVELS="0 0.1 0.2 0.5 1 2 3 4 5")
            self.add(f"noise-{self.name}-{f}", ["bash", "experiments/l2s_noise_robustness/run.sh"],
                [out / "mc1_by_noise.csv"], env)

    def bipo(self):
        if self.args.model != "llama":
            raise ValueError("The archived BiPO adaptation is Llama layer 14 only; use --model llama")
        if not self.args.model_dir:
            raise ValueError("BiPO uses local_files_only: supply --model-dir with downloaded Llama weights")
        self.prepare(train=False)
        out = self.work / "bipo" / self.name
        self.add("bipo-llama", self.py("experiments/bipo_truthfulqa/run.py", "--model-dir", self.model_dir,
            "--feature", self.feature, "--output", out, "--folds", *self.args.folds,
            "--seed", self.args.seed, "--epochs", self.args.bipo_epochs), [out / "performance.json"])

    def generation(self):
        self.prepare(l2s=True, prototypes=True)
        for f in self.args.folds:
            out = self.work / "generation" / self.name / f"fold{f}"
            self.add(f"generation-{self.name}-{f}", self.py("reproducibility/generation.py",
                "--model", self.model_dir, "--model-name", self.name, "--layer", self.layer,
                "--s2s-checkpoint", self.dyn(f), "--l2s-checkpoint", self.l2s(f),
                "--prompts", self.args.prompts or "data/mmsafe.jsonl", "--output-dir", out,
                "--dtype", self.args.dtype, "--device", self.args.device,
                "--max-new-tokens", 200, "--seed", self.args.seed), [out / "generations.jsonl"])


def build_plan(args):
    if args.suite == "scaling":
        jobs = []
        for key in args.scaling_models:
            out = args.work_dir / "scaling" / key / ("nf4" if key == "qwen14" else "bfloat16")
            cmd = [sys.executable, "experiments/model_scaling_ablation/run.py", "--model", key,
                   "--output-dir", str(out), "--epochs", str(args.epochs), "--seed", str(args.seed)]
            if args.model_root:
                from experiments.model_scaling_ablation.common import MODELS as entries
                cmd += ["--model-dir", str(args.model_root / entries[key][0].split("/")[-1])]
            if key == "qwen14":
                cmd += ["--load-in-4bit"]
            jobs.append(dict(name=f"scaling-{key}", command=cmd, outputs=[str(out / "summary.json")], env={}))
        return jobs
    plan = Plan(args)
    suites = ("main", "generic", "component", "gate", "margin", "noise", "generation") if args.suite == "all" else (args.suite,)
    for suite in suites:
        getattr(plan, suite)()
    return plan.jobs


def execute(jobs, args):
    if args.suite in ("generation", "all"):
        from reproducibility.generation import read_prompts
        read_prompts(args.prompts or "data/mmsafe.jsonl")
    if args.suite != "scaling" and (not args.model_dir or not Path(args.model_dir).is_dir()):
        raise ValueError("Execution requires --model-dir with downloaded weights; use tools/download_model.py first")
    versions = {}
    for package in ("torch", "transformers", "datasets", "numpy", "pandas", "baukit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("Missing runtime dependency: " + package)
    model_files = {}
    if args.model_dir:
        model_dir = Path(args.model_dir)
        model_files = {x.name: digest(x) for x in model_dir.iterdir() if x.suffix in (".json", ".model", ".jinja") and x.is_file()}
    fingerprint = source_digest()
    for job in jobs:
        print("\n[" + job["name"] + "] " + shlex.join(job["command"]), flush=True)
        receipt = args.work_dir / "receipts" / (job["name"] + ".json")
        context = dict(job=job, source_sha256=fingerprint, versions=versions, python=platform.python_version(), model_files=model_files)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved.get("context") != context:
                raise ValueError(f"Configuration/source changed: choose a new --work-dir ({job['name']})")
            if args.resume and all(Path(p).is_file() and digest(p) == h for p, h in saved["outputs"].items()):
                print("  reusing verified outputs", flush=True)
                continue
        elif any(Path(p).exists() for p in job["outputs"]):
            raise ValueError(f"Unverified pre-existing outputs: choose a new --work-dir ({job['name']})")
        inherited = dict(os.environ)
        if job["command"][0] == "bash":
            # Native shell configs use ${VAR:-default}. Clear implicit overrides
            # so the recorded explicit environment controls this reproduction.
            keys = ("MODEL MODEL_DIR MODEL_LABEL LLAMA_MODEL QWEN_MODEL LLAMA_LAYER QWEN_LAYER "
                    "LAYER FOLD_IDX NUM_FOLDS VAL_FRACTION MLP_HIDDEN BATCH_SIZE LEARNING_RATE "
                    "ALPHA BETA KAPPA STRENGTH SLERP_T WEIGHT_KAPPA MARGIN METHOD POSITIVE_MODE "
                    "FEW_SHOT NO_INSTRUCTION NO_INTERVENTION EVAL_SPLIT EVAL_PRESET MAX_SAMPLES "
                    "RUN_TRAIN RUN_EXTRACT RUN_EVAL RUN_PROTOTYPE DRY_RUN FEATURE_PATH OUTPUT_PATH "
                    "EXTRA_ARGS RUN_TRAIN NOISE_LEVELS NOISE_SEED EXTRACT_BATCH_SIZE").split()
            for key in keys:
                inherited.pop(key, None)
        env = {**inherited, "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONPATH": str(ROOT),
               "PYTHONUNBUFFERED": "1", **job["env"]}
        subprocess.run(job["command"], cwd=ROOT, env=env, check=True)
        missing = [p for p in job["outputs"] if not Path(p).is_file()]
        if missing:
            raise RuntimeError("Missing expected outputs: " + repr(missing))
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(dict(context=context, outputs={p: digest(p) for p in job["outputs"]}), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=["main", "generic", "component", "gate", "margin", "noise", "scaling", "bipo", "generation", "all"])
    p.add_argument("--model", choices=MODELS, default="llama")
    p.add_argument("--datasets", nargs="+", choices=["boolq", "copa", "storycloze", "mmlu_global", "winogrande"],
                   default=["boolq", "copa", "storycloze", "mmlu_global", "winogrande"])
    p.add_argument("--model-dir", help="Local model directory, or official Hub ID except for BiPO")
    p.add_argument("--model-root", type=Path)
    p.add_argument("--scaling-models", nargs="+", choices=SCALING, default=list(SCALING))
    p.add_argument("--folds", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--work-dir", type=Path, default=Path("results/reproduction"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   help="Default FP16; scaling is fixed to BF16/NF4")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--bipo-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts", help="MMsafe text prompt JSONL; not bundled with the original source")
    p.add_argument("--execute", action="store_true", help="Actually load models and run; default only prints JSON plan")
    p.add_argument("--resume", action="store_true", help="Reuse matching receipts and verified output hashes")
    args = p.parse_args()
    args.dtype = args.dtype or ("bfloat16" if args.suite == "scaling" else "float16")
    if len(set(args.folds)) != len(args.folds) or args.epochs <= 0 or args.bipo_epochs <= 0:
        p.error("folds must be unique and epochs positive")
    if args.suite == "scaling" and (args.device != "cuda" or args.dtype != "bfloat16"):
        p.error("The declared scaling comparison uses CUDA BF16 (14B: NF4 with BF16 compute)")
    if args.suite in ("gate", "margin", "bipo") and args.device != "cuda":
        p.error("The archived gate/margin/BiPO evaluators require CUDA")
    if args.suite == "bipo" and args.dtype != "float16":
        p.error("The archived BiPO adaptation is FP16; choose a separate experiment for other precision")
    if args.suite in ("main", "generic", "all") and (args.dtype != "float16" or args.device != "cuda"):
        p.error("The historical Spherical evaluator is hard-coded FP16; main/all require CUDA FP16")
    try:
        jobs = build_plan(args)
        if args.execute:
            execute(jobs, args)
        else:
            print(json.dumps(dict(suite=args.suite, mode="plan_only", jobs=jobs), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
