"""Prepare each layer/fold once, evaluate both grids, and select fold-mean MC1."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

from results import cells, file_identity, fingerprint, write_json, write_summaries

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def env(name, default):
    return os.environ.get(name, str(default))


def numbers(text):
    values = [float(x) for x in text.replace(",", " ").split()]
    if not values or len(set(values)) != len(values) or not all(map(math.isfinite, values)):
        raise argparse.ArgumentTypeError("Grid values must be nonempty, finite and unique")
    return values


def layers(text):
    try:
        values = []
        for item in text.replace(",", " ").split():
            if "-" in item:
                start, end = map(int, item.split("-"))
                values.extend(range(start, end + 1))
            else:
                values.append(int(item))
        if not values or len(set(values)) != len(values) or any(x < 0 or x > 31 for x in values):
            raise ValueError()
        return sorted(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use unique Llama layers 0..31, e.g. '0-31' or '0 14 31'") from exc


def cached(path, context):
    try:
        stamp = json.loads(Path(str(path) + ".cache.json").read_text())
        return stamp == {"context": context, "artifact": file_identity(path)}
    except (OSError, ValueError):
        return False


def stamp(path, context):
    write_json(str(path) + ".cache.json", {"context": context, "artifact": file_identity(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=layers, default=env("LAYERS", "0-31"))
    parser.add_argument("--methods", nargs="+", choices=["S2S-L", "S2S-R"],
                        default=env("METHODS", "S2S-L S2S-R").split())
    parser.add_argument("--model-dir", default=env("MODEL_DIR", "models/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--output-dir", type=Path, default=env("OUTPUT_DIR", ROOT / "results/experiment/rebuttal_R2W5_layer_sweep/LLAMA"))
    for name, default in (("strength", "0.5 1 1.5 2 2.5 3"), ("slerp_t", "0 0.2 0.4 0.6 0.8 1"),
                          ("weight_kappa", "1 2 3 5 10 20"), ("margin", "-0.9 -0.7 -0.5 -0.3 0 0.3")):
        parser.add_argument("--" + name.replace("_", "-"), type=numbers, default=env(name.upper() + "_VALUES", default))
    parser.add_argument("--dry-run", action="store_true", default=env("DRY_RUN", 0) == "1")
    args = parser.parse_args()
    if not args.methods or len(set(args.methods)) != len(args.methods) or not set(args.methods) <= {"S2S-L", "S2S-R"}:
        parser.error("methods must be unique S2S-L/S2S-R values")
    if any(x < 0 for x in args.strength) or any(not 0 <= x <= 1 for x in args.slerp_t) or any(x <= 0 for x in args.weight_kappa):
        parser.error("Require strength >= 0, slerp_t in [0,1], weight_kappa > 0")
    if env("NUM_FOLDS", 2) != "2" or set(env("FOLDS", "0 1").split()) != {"0", "1"} or env("EVAL_SPLIT", "test") != "test":
        parser.error("This experiment requires folds 0/1, NUM_FOLDS=2 and EVAL_SPLIT=test")
    for name in ("RUN_EXTRACT", "RUN_TRAIN"):
        if env(name, "auto") not in {"auto", "0", "1"}:
            parser.error(f"{name} must be auto, 0 or 1")
    if env("REUSE_EXISTING", 1) not in {"0", "1"}:
        parser.error("REUSE_EXISTING must be 0 or 1")
    training = {"positive_mode": env("POSITIVE_MODE", "mean"), "num_folds": 2,
                "val_fraction": float(env("VAL_FRACTION", 0.5)), "mlp_hidden": int(env("MLP_HIDDEN", 256)),
                "epochs": int(env("EPOCHS", 2000)), "batch_size": int(env("BATCH_SIZE", 256)),
                "learning_rate": float(env("LEARNING_RATE", 0.001)), "seed": int(env("SEED", 42)),
                "device": env("DEVICE", "cuda")}
    # Include implementation content in cache identities, including shared method code.
    sources = sorted(set(list(HERE.glob("*.py")) + list((ROOT / "methods/S2S_R").glob("*.py"))
                         + list((ROOT / "methods/S2S_L").glob("*.py")) + list((ROOT / "methods").glob("*.py"))
                         + [ROOT / "truthfulqa/models.py"]))
    code = fingerprint({str(p.relative_to(ROOT)): p.read_text() for p in sources})
    config = {"model_name": env("MODEL", "llama3.1-8B-Instruct"), "model_dir": str(Path(args.model_dir).resolve()),
              "device": training["device"], "dtype": env("DTYPE", "float16"), "seed": training["seed"],
              "methods": args.methods, "layers": args.layers, "training": training, "code": code,
              "grid": {name: getattr(args, name) for name in ("strength", "slerp_t", "weight_kappa", "margin")}}
    if config["device"] not in {"cuda", "cpu"} or config["dtype"] not in {"float16", "bfloat16", "float32"}:
        parser.error("Unsupported DEVICE or DTYPE")
    if training["positive_mode"] not in {"mean", "mc1"} or not 0 < training["val_fraction"] < 1:
        parser.error("Require POSITIVE_MODE=mean/mc1 and 0 < VAL_FRACTION < 1")
    if any(not math.isfinite(training[k]) or training[k] <= 0 for k in ("mlp_hidden", "epochs", "batch_size", "learning_rate")):
        parser.error("MLP dimensions and optimizer settings must be positive")
    output = args.output_dir.resolve()
    config_path = output / "run_config.json"
    if not args.dry_run:
        write_json(config_path, config)
    def run(command):
        print(("[dry-run] " if args.dry_run else "[run] ") + shlex.join(map(str, command)), flush=True)
        if not args.dry_run:
            subprocess.run(list(map(str, command)), check=True, cwd=ROOT)
    reports = []
    count = sum(1 for layer in args.layers for _ in cells(config, layer))
    print(f"Llama layers={args.layers}; methods={args.methods}; fold-specific evaluations={count}; selection=test_set_sweep_best", flush=True)
    for layer in args.layers:
        feature = output / "features" / f"layer{layer}.npz"
        feature_context = {k: config[k] for k in ("model_name", "model_dir", "device", "dtype", "code")}
        feature_context["layer"] = layer
        mode = env("RUN_EXTRACT", "auto")
        valid = not args.dry_run and cached(feature, feature_context)
        if mode == "1" or not valid:
            if mode == "0" and not args.dry_run:
                raise ValueError(f"Missing/mismatched feature cache {feature}; use RUN_EXTRACT=auto")
            run([sys.executable, HERE / "extract_features.py", "--config", config_path, "--layer", layer, "--output", feature])
            if not args.dry_run:
                stamp(feature, feature_context)
        checkpoints = []
        for fold in (0, 1):
            checkpoint = output / "checkpoints" / f"layer{layer}" / f"fold{fold}.npz"
            context = {"feature": str(feature) if args.dry_run else file_identity(feature), "training": training, "fold": fold, "code": code}
            mode = env("RUN_TRAIN", "auto")
            valid = not args.dry_run and cached(checkpoint, context)
            if mode == "1" or not valid:
                if mode == "0" and not args.dry_run:
                    raise ValueError(f"Missing/mismatched checkpoint {checkpoint}; use RUN_TRAIN=auto")
                command = [sys.executable, ROOT / "methods/S2S_R/train_predictor.py", "--feature_path", feature,
                           "--output_path", checkpoint, "--fold_idx", fold]
                for name, value in training.items():
                    command.extend(["--" + name, value])
                run(command)
                if not args.dry_run:
                    stamp(checkpoint, context)
            checkpoints.append(checkpoint)
        command = [sys.executable, HERE / "evaluate_grid.py", "--config", config_path, "--layer", layer,
                   "--checkpoint0", checkpoints[0], "--checkpoint1", checkpoints[1], "--output", output]
        if env("REUSE_EXISTING", 1) == "1":
            command.append("--resume")
        run(command)
        if not args.dry_run:
            for _, _, _, relative in cells(config, layer):
                reports.append(json.loads((output / relative).read_text()))
            write_summaries(output, reports, config)
    print(f"{'Planned outputs' if args.dry_run else 'Saved summaries'}: {output / 'summaries'}", flush=True)


if __name__ == "__main__":
    main()
