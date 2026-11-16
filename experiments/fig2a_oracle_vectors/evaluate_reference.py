"""MC1 comparison of exact answer-pair vectors and their shared mean."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fig2a_oracle_vectors.vector_utils import (  # noqa: E402
    find_answer_rows,
    load_pair_vector_data,
)
from method.S2S_R.eval_utils import (  # noqa: E402
    HF_NAMES,
    calculate_scores_zeroshot,
    get_additive_intervention,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
)

METHOD_LABELS = {
    "pair_specific": "Answer-pair vector",
    "shared_mean": "Shared mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 2(a) oracle diagnostic: exact per-answer V+ - V- versus their shared mean"
    )
    parser.add_argument("model_name", nargs="?", default="llama3.1-8B-Instruct")
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--fold_idx", type=int, choices=[0,1], default=0)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Use 1.0 for the exact h- + (h+ - h-) construction.",
    )
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--save_details", action="store_true")
    args = parser.parse_args()
    if args.alpha == 0:
        parser.error("--alpha must be nonzero")
    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max_questions must be positive")
    return args


def score_one_choice(
    *,
    model,
    tokenizer,
    question: str,
    answer: str,
    vector: np.ndarray | None,
    alpha: float,
    layer_name: str,
    device: torch.device,
    use_instruction: bool,
    model_name: str,
) -> float:
    hook = None
    if vector is not None:
        vector_tensor = torch.as_tensor(vector, dtype=torch.float32, device=device)
        hook = get_additive_intervention(vector_tensor, alpha)
    return calculate_scores_zeroshot(
        model=model,
        tokenizer=tokenizer,
        question=question,
        choices=[answer],
        base_hook_fn=hook,
        layer_name=layer_name,
        device=device,
        use_instruction=use_instruction,
        model_name=model_name,
    )[0]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    pair_data = load_pair_vector_data(args.feature_path, fold_idx=args.fold_idx)
    print(
        f"fold={pair_data.fold_idx} | train questions={len(pair_data.train_questions)} | "
        f"train answer pairs={pair_data.num_train_pairs} | alpha={args.alpha}"
    )
    print(
        f"mean ||v_ij||={pair_data.mean_pair_norm:.6f} | "
        f"||mean(v_ij)||={pair_data.mean_vector_norm:.6f} | "
        f"max exact-reconstruction error={pair_data.max_reconstruction_error:.3e}"
    )

    # Validate all question and answer mappings before loading the model.
    _, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    selected_ids = pair_data.train_questions.tolist()
    if args.max_questions is not None:
        selected_ids = selected_ids[: args.max_questions]
    answer_rows: dict[int, tuple[int, list[int]]] = {}
    for question_id in selected_ids:
        mc1_targets = hf_dataset[int(question_id)]["mc1_targets"]
        mc1_choices = list(mc1_targets["choices"])
        mc1_labels = list(mc1_targets["labels"])
        correct_answers = [
            choice for choice, label in zip(mc1_choices, mc1_labels) if label == 1
        ]
        false_answers = [
            choice for choice, label in zip(mc1_choices, mc1_labels) if label == 0
        ]
        if len(correct_answers) != 1:
            raise ValueError(
                f"Question {question_id}: MC1 needs exactly one correct answer, "
                f"found {len(correct_answers)}"
            )
        answer_rows[question_id] = find_answer_rows(
            pair_data, question_id, correct_answers[0], false_answers
        )

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    layer_name = get_layer_name(args.model_name, args.layer)

    details = {method: [] for method in METHOD_LABELS}
    iterator = tqdm(selected_ids, desc="MC1 answer-pair vs mean")
    for question_id in iterator:
        positive_row, false_rows = answer_rows[question_id]
        question = str(hf_dataset[question_id]["question"])
        best_answer = pair_data.choices[positive_row]

        # The MC1 correct answer h+ is always scored without intervention.
        correct_score = score_one_choice(
            model=model,
            tokenizer=tokenizer,
            question=question,
            answer=best_answer,
            vector=None,
            alpha=args.alpha,
            layer_name=layer_name,
            device=device,
            use_instruction=not args.no_instruction,
            model_name=args.model_name,
        )

        for method in METHOD_LABELS:
            false_scores = []
            for false_row in false_rows:
                vector = (
                    pair_data.vectors[(positive_row, false_row)]
                    if method == "pair_specific"
                    else pair_data.shared_mean
                )
                false_scores.append(
                    score_one_choice(
                        model=model,
                        tokenizer=tokenizer,
                        question=question,
                        answer=pair_data.choices[false_row],
                        vector=vector,
                        alpha=args.alpha,
                        layer_name=layer_name,
                        device=device,
                        use_instruction=not args.no_instruction,
                        model_name=args.model_name,
                    )
                )
            max_false_score = float(max(false_scores))
            details[method].append(
                {
                    "hf_idx": question_id,
                    "question": question,
                    "best_answer": best_answer,
                    "correct_score_no_intervention": float(correct_score),
                    "max_false_score": max_false_score,
                    "margin": float(correct_score - max_false_score),
                    "MC1": int(correct_score > max_false_score),
                    "num_false_answers": len(false_rows),
                }
            )
        iterator.set_postfix(
            {
                method: f"{np.mean([item['MC1'] for item in rows]):.3f}"
                for method, rows in details.items()
            }
        )

    metrics = {
        method: float(np.mean([item["MC1"] for item in rows]))
        for method, rows in details.items()
    }
    payload = {
        "experiment": "fig1_train_answer_pair_vectors_vs_shared_mean_mc1",
        "definition": {
            "pair_specific": (
                "For each false answer j, use its exact raw vector "
                "v_ij = h_i+ - h_j- to the MC1 best truthful answer i; "
                "do not average or normalize."
            ),
            "shared_mean": (
                "Replace every v_ij by the arithmetic mean of all raw v_ij "
                "from the outer-fold training questions; do not normalize."
            ),
            "correct_answer": "Score the MC1 best truthful answer without intervention.",
            "evaluation_split": "outer-fold training questions",
        },
        "model_name": args.model_name,
        "model_path": model_path,
        "feature_path": args.feature_path,
        "layer": args.layer,
        "alpha": args.alpha,
        "seed": args.seed,
        "fold_idx": pair_data.fold_idx,
        "num_eval_questions": len(selected_ids),
        "num_train_pairs": pair_data.num_train_pairs,
        "geometry": {
            "mean_pair_vector_norm": pair_data.mean_pair_norm,
            "shared_mean_vector_norm": pair_data.mean_vector_norm,
            "max_exact_reconstruction_error": pair_data.max_reconstruction_error,
        },
        "methods": {
            method: {"label": METHOD_LABELS[method], "metrics": {"MC1": value}}
            for method, value in metrics.items()
        },
        "shared_mean_minus_pair_specific": {
            "MC1": metrics["shared_mean"] - metrics["pair_specific"]
        },
    }
    output_path = Path(args.output_path)
    write_json(output_path, payload)
    if args.save_details:
        for method, rows in details.items():
            detail_path = output_path.with_name(f"{output_path.stem}_{method}_details.csv")
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            print(f"Details: {detail_path}")
    print(f"Answer-pair vector MC1: {metrics['pair_specific']:.6f}")
    print(f"Shared mean MC1:       {metrics['shared_mean']:.6f}")
    print(f"Delta:                 {metrics['shared_mean'] - metrics['pair_specific']:+.6f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
