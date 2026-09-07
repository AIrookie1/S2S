"""Evaluate Fig. 1's two MC1 conditions for every available model layer.

The feature files are the same pair-vector NPZs consumed by Fig. 1.  One model
is loaded and reused across layers; after each layer finishes, a small JSON
containing only the two MC1 values is atomically written to ``--output_dir``.
This makes long runs resumable and intentionally does not write per-question
details or figures.
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

from experiments.fig1_train_mean.evaluate_train_vectors import score_one_choice
from experiments.fig1_train_mean.vector_utils import find_answer_rows, load_pair_vector_data
from method.dyn3d_sga.eval_utils import (
    HF_NAMES,
    calculate_scores_zeroshot,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)


METHODS = ("pair_specific", "shared_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fig. 1a: pair-specific versus shared-mean MC1 for every layer"
    )
    parser.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    parser.add_argument(
        "--feature_template",
        default="./features/llama3.1-8B-Instruct_layer{layer}_dynSVD_pairs.npz",
        help="Feature path template containing {layer} and {fold}.",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Layers to evaluate. If omitted, discover layers from feature files.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--output_dir", default="./experiments/fig1a/results")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip a layer when one of its fold feature files is missing.",
    )
    args = parser.parse_args()
    if args.alpha == 0:
        parser.error("--alpha must be nonzero")
    if not args.folds or len(set(args.folds)) != len(args.folds):
        parser.error("--folds must contain at least one unique fold")
    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max_questions must be positive")
    return args


def discover_layers(template: str, folds: list[int]) -> list[int]:
    """Find layer numbers from a template while requiring at least one fold."""
    if "{layer}" not in template:
        raise ValueError("--feature_template must contain {layer}")
    probe = template.replace("{layer}", "*").replace("{fold}", "*")
    found: set[int] = set()
    for path in glob.glob(probe):
        match = re.search(r"layer(\d+)", Path(path).name)
        if match:
            found.add(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No feature files matched template: {probe}")
    return sorted(found)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_model(args: argparse.Namespace, device: torch.device):
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device.type == "cpu" else {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, model_path


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
    for pair_data in fold_data:
        selected_ids = pair_data.train_questions.tolist()
        if args.max_questions is not None:
            selected_ids = selected_ids[: args.max_questions]
        answer_rows: dict[int, tuple[int, list[int]]] = {}
        for question_id in selected_ids:
            targets = hf_dataset[int(question_id)]["mc1_targets"]
            choices = list(targets["choices"])
            labels = list(targets["labels"])
            correct = [c for c, label in zip(choices, labels) if label == 1]
            false = [c for c, label in zip(choices, labels) if label == 0]
            if len(correct) != 1:
                raise ValueError(f"Question {question_id}: MC1 needs exactly one correct answer")
            answer_rows[question_id] = find_answer_rows(pair_data, question_id, correct[0], false)

        for question_id in tqdm(selected_ids, desc=f"layer {layer} fold {pair_data.fold_idx}", leave=False):
            positive_row, false_rows = answer_rows[question_id]
            question = str(hf_dataset[question_id]["question"])
            best_answer = pair_data.choices[positive_row]
            correct_score = score_one_choice(
                model=model, tokenizer=tokenizer, question=question, answer=best_answer,
                vector=None, alpha=args.alpha, layer_name=layer_name, device=device,
                use_instruction=not args.no_instruction, model_name=args.model_name,
            )
            for method in METHODS:
                false_scores = []
                for false_row in false_rows:
                    vector = pair_data.vectors[(positive_row, false_row)] if method == "pair_specific" else pair_data.shared_mean
                    false_scores.append(score_one_choice(
                        model=model, tokenizer=tokenizer, question=question,
                        answer=pair_data.choices[false_row], vector=vector,
                        alpha=args.alpha, layer_name=layer_name, device=device,
                        use_instruction=not args.no_instruction, model_name=args.model_name,
                    ))
                totals[method] += int(correct_score > max(false_scores))
            count += 1
    if count == 0:
        raise ValueError(f"Layer {layer}: no evaluation questions")
    return {
        "layer": layer,
        "alpha": args.alpha,
        "model_name": args.model_name,
        "folds": [int(x) for x in args.folds],
        "num_questions": count,
        "MC1": {method: float(totals[method] / count) for method in METHODS},
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    layers = sorted(set(args.layers)) if args.layers is not None else discover_layers(args.feature_template, args.folds)
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("layers must be non-negative")
    _, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    model, tokenizer, model_path = load_model(args, device)
    print(f"Fig. 1a | layers={layers} | folds={args.folds} | model={model_path}")
    for layer in layers:
        paths = [Path(args.feature_template.format(layer=layer, fold=fold)) for fold in args.folds]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            message = f"Layer {layer} missing feature files: {missing}"
            if args.allow_missing:
                print(f"WARNING: {message}; skipping")
                continue
            raise FileNotFoundError(message)
        fold_data = [load_pair_vector_data(str(path)) for path in paths]
        payload = evaluate_layer(layer, args, fold_data, hf_dataset, model, tokenizer, device)
        payload["model_path"] = model_path
        output_path = Path(args.output_dir) / f"layer{layer}_mc1.json"
        write_json(output_path, payload)
        print(f"Saved layer {layer}: {output_path} | {payload['MC1']}")


if __name__ == "__main__":
    main()
