"""Layer-wise comparison of raw-vs-mean and linear-vs-spherical steering.

Four conditions are evaluated on the same training-fold MC1 questions:
  1. linear + exact answer-pair vector
  2. linear + shared mean vector
  3. spherical + exact answer-pair vector
  4. spherical + shared mean vector

Each layer uses exactly one base ``*_dynSVD_pairs.npz`` file.  Fold 0/1
membership is reconstructed in memory from that file; no fold-specific feature
NPZ files are required.

Every scored choice is intervened on.  For an exact answer-pair vector v_ij,
the truthful and false candidate in that pair are both scored with the same
intervention.  MC1 is one only when the truthful answer wins every pairwise
comparison.  For the shared mean, one common intervention is used for the
truthful answer and all false candidates.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.steering_vector_ablation.vector_utils import (  # noqa: E402
    find_answer_rows,
    load_pair_vector_data,
)
from method.dyn3d_sga.eval_utils import (  # noqa: E402
    HF_NAMES,
    calculate_scores_zeroshot,
    get_additive_intervention,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)
from method.spherical_steering.steering import get_spherical_intervention  # noqa: E402


METHODS = (
    "linear_pair_specific",
    "linear_shared_mean",
    "spherical_pair_specific",
    "spherical_shared_mean",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steering-vector ablation across layers")
    p.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    p.add_argument(
        "--feature_template",
        required=True,
        help=(
            "Single base pair-feature path template containing {layer}, e.g. "
            "features/model_layer{layer}_dynSVD_pairs.npz. Do not include {fold}."
        ),
    )
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    p.add_argument("--linear_alpha", type=float, default=1.0)
    p.add_argument("--spherical_kappa", type=float, default=20.0)
    p.add_argument("--spherical_alpha", type=float, required=True)
    p.add_argument("--spherical_beta", type=float, required=True)
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument(
        "--output_dir",
        default="./results/experiment/steering_vector_ablation",
    )
    p.add_argument("--allow_missing", action="store_true")
    args = p.parse_args()

    if args.linear_alpha == 0:
        p.error("--linear_alpha must be nonzero")
    if args.num_folds < 2:
        p.error("--num_folds must be >= 2")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        p.error("--folds must contain unique values")
    if any(fold < 0 or fold >= args.num_folds for fold in args.folds):
        p.error("each --folds value must satisfy 0 <= fold < num_folds")
    if "{fold}" in args.feature_template:
        p.error(
            "--feature_template must point to the single *_dynSVD_pairs.npz file; "
            "remove the {fold} placeholder"
        )
    return args


def discover_layers(template: str) -> list[int]:
    probe = template.replace("{layer}", "*")
    found: set[int] = set()
    for path in glob.glob(probe):
        match = re.search(r"layer(\d+)", Path(path).name)
        if match:
            found.add(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No feature files matched {probe}")
    return sorted(found)


def render_feature_path(template: str, layer: int) -> Path:
    try:
        rendered = template.format(layer=layer)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid --feature_template {template!r}; expected only {{layer}}"
        ) from exc
    return Path(rendered)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def normalize_vector(vector: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(vector, dtype=torch.float32, device=device).flatten()
    norm = torch.linalg.norm(tensor)
    if norm.item() == 0:
        raise ValueError("Steering vector has zero norm")
    return tensor / norm


def make_hook(
    kind: str,
    vector: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    if kind == "linear":
        return get_additive_intervention(
            torch.as_tensor(vector, dtype=torch.float32, device=device),
            args.linear_alpha,
        )

    mu_t = normalize_vector(vector, device)
    mu_h = -mu_t
    return get_spherical_intervention(
        mu_t,
        mu_h,
        args.spherical_kappa,
        args.spherical_alpha,
        args.spherical_beta,
    )


def score(
    model,
    tokenizer,
    question: str,
    answer: str,
    hook,
    layer_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    return calculate_scores_zeroshot(
        model,
        tokenizer,
        question,
        [answer],
        hook,
        layer_name,
        device,
        use_instruction=not args.no_instruction,
        model_name=args.model_name,
    )[0]


def load_model(args: argparse.Namespace, device: torch.device):
    path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.float32
        if device.type == "cpu"
        else {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, path


def evaluate_layer(
    layer: int,
    args: argparse.Namespace,
    fold_data: list,
    hf_dataset,
    model,
    tokenizer,
    device: torch.device,
) -> dict:
    totals = {method: 0 for method in METHODS}
    count = 0
    layer_name = get_layer_name(args.model_name, layer)

    for data in fold_data:
        ids = data.train_questions.tolist()
        if args.max_questions:
            ids = ids[: args.max_questions]

        mappings = {}
        for question_id in ids:
            targets = hf_dataset[int(question_id)]["mc1_targets"]
            choices = list(targets["choices"])
            labels = list(targets["labels"])
            correct = [
                choice for choice, label in zip(choices, labels) if label == 1
            ]
            false = [
                choice for choice, label in zip(choices, labels) if label == 0
            ]
            if len(correct) != 1:
                raise ValueError(
                    f"Question {question_id}: expected exactly one MC1 truthful answer"
                )
            mappings[question_id] = find_answer_rows(
                data,
                question_id,
                correct[0],
                false,
            )

        for question_id in tqdm(
            ids,
            desc=f"layer {layer} fold {data.fold_idx}",
            leave=False,
        ):
            positive_row, false_rows = mappings[question_id]
            question = str(hf_dataset[question_id]["question"])
            truth = data.choices[positive_row]
            mean_vector = data.shared_mean

            # Shared mean: use the same intervention for every choice.
            for kind, name in (
                ("linear", "linear_shared_mean"),
                ("spherical", "spherical_shared_mean"),
            ):
                hook = make_hook(kind, mean_vector, args, device)
                truth_score = score(
                    model,
                    tokenizer,
                    question,
                    truth,
                    hook,
                    layer_name,
                    device,
                    args,
                )
                false_scores = [
                    score(
                        model,
                        tokenizer,
                        question,
                        data.choices[row],
                        hook,
                        layer_name,
                        device,
                        args,
                    )
                    for row in false_rows
                ]
                totals[name] += int(truth_score > max(false_scores))

            # Exact pair vector: for each false candidate, score both members
            # of that pair with that pair's raw v_ij intervention.
            for kind, name in (
                ("linear", "linear_pair_specific"),
                ("spherical", "spherical_pair_specific"),
            ):
                wins = []
                for row in false_rows:
                    vector = data.vectors[(positive_row, row)]
                    hook = make_hook(kind, vector, args, device)
                    truth_score = score(
                        model,
                        tokenizer,
                        question,
                        truth,
                        hook,
                        layer_name,
                        device,
                        args,
                    )
                    false_score = score(
                        model,
                        tokenizer,
                        question,
                        data.choices[row],
                        hook,
                        layer_name,
                        device,
                        args,
                    )
                    wins.append(truth_score > false_score)
                totals[name] += int(all(wins))

            count += 1

    if count == 0:
        raise ValueError(f"Layer {layer}: no questions")

    return {
        "layer": layer,
        "model_name": args.model_name,
        "folds": [int(x) for x in args.folds],
        "num_folds": int(args.num_folds),
        "num_questions": count,
        "fold_storage": "single_base_pairs_npz_in_memory_split",
        "parameters": {
            "linear_alpha": args.linear_alpha,
            "spherical_kappa": args.spherical_kappa,
            "spherical_alpha": args.spherical_alpha,
            "spherical_beta": args.spherical_beta,
        },
        "MC1": {
            method: float(totals[method] / count)
            for method in METHODS
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    layers = (
        sorted(set(args.layers))
        if args.layers is not None
        else discover_layers(args.feature_template)
    )

    _, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    model, tokenizer, model_path = load_model(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Steering vector ablation | "
        f"model={model_path} | layers={layers} | folds={args.folds} | "
        "folds=derived in memory from one pairs.npz per layer"
    )

    for layer in layers:
        feature_path = render_feature_path(args.feature_template, layer)
        if not feature_path.is_file():
            if args.allow_missing:
                print(f"WARNING: layer {layer} missing {feature_path}; skipped")
                continue
            raise FileNotFoundError(
                f"Layer {layer} missing base pair feature: {feature_path}"
            )

        fold_data = [
            load_pair_vector_data(
                str(feature_path),
                fold_idx=fold,
                num_folds=args.num_folds,
            )
            for fold in args.folds
        ]

        payload = evaluate_layer(
            layer,
            args,
            fold_data,
            hf_dataset,
            model,
            tokenizer,
            device,
        )
        payload["model_path"] = model_path
        payload["feature_path"] = str(feature_path)
        output_path = (
            output_dir / f"{args.model_name}_layer{layer}_mc1.json"
        )
        write_json(output_path, payload)
        print(f"layer {layer}: {payload['MC1']}")


if __name__ == "__main__":
    main()
