"""Validated, dependency-free configuration loading for reproduction commands."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_profile(path=None, model=None):
    path = Path(path) if path else ROOT / "configs/reproduction" / ((model or "llama") + ".json")
    if not path.is_absolute():
        path = ROOT / path
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1 or profile.get("model") not in ("llama", "qwen"):
        raise ValueError("Unsupported model configuration")
    if model is not None and model != profile["model"]:
        raise ValueError("--model and --config refer to different model families")
    if not isinstance(profile.get("layer"), int) or profile["layer"] < 0:
        raise ValueError("The intervention layer must be a nonnegative integer")
    training = profile["training"]
    if training["num_folds"] != 2 or not 0 < training["predictor_fraction"] < 1:
        raise ValueError("The main protocol requires two folds and a valid predictor fraction")
    for key in ("mlp_hidden", "epochs", "batch_size", "learning_rate"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError("Invalid training parameter: " + key)
    if set(profile["main"]["s2s_r"]["margins"]) != {"0", "1"}:
        raise ValueError("Declare an S2S_R margin for both folds")
    profile["config_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return profile


def main_parameters(profile, fold):
    values = copy.deepcopy(profile["main"])
    values["s2s_r"]["margin"] = values["s2s_r"].pop("margins")[str(fold)]
    return values


def load_registry():
    path = ROOT / "configs/experiments.json"
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported experiment registry")
    entries = registry["experiments"]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Experiment IDs must be unique")
    for entry in entries:
        if entry.get("entry") and not (ROOT / entry["entry"]).is_file():
            raise ValueError("Missing experiment entry: " + entry["entry"])
    return {entry["id"]: entry for entry in entries}
