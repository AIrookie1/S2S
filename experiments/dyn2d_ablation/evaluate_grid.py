"""Run the requested 6x6 Dyn2D TruthfulQA MC ablation in one model process."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn2d.steering import get_2d_dynamic_intervention
from method.dyn3d_sga.evaluate_dyn_mlp import load_components
from method.dyn3d_sga.eval_utils import (
    HF_NAMES,
    calculate_scores_fewshot,
    calculate_scores_zeroshot,
    format_best,
    get_layer_name,
    load_truthfulqa_mc_dataframe,
    resolve_device,
    set_seed,
    split_multi_answer,
)
from truthfulqa.models import MC_calcs, set_columns

DEFAULT_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_SLERPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _parse_float_list(text: str) -> list[float]:
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    if not all(np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def evaluate_one(
    *,
    model,
    tokenizer,
    base_dataframe,
    model_name: str,
    layer_name: str,
    device: torch.device,
    avg: torch.Tensor,
    predictor,
    strength: float,
    slerp_t: float,
    few_shot: bool,
    preset: str,
    no_instruction: bool,
) -> tuple[dict[str, float], dict[str, float | int]]:
    results = base_dataframe.copy().reset_index(drop=True)
    tag = model_name
    set_columns(tag, results)
    stats = {"total": 0, "steered": 0, "prediction_count": 0}
    hook = get_2d_dynamic_intervention(
        avg,
        predictor,
        strength=strength,
        slerp_t=slerp_t,
        stats=stats,
    )

    progress = tqdm(
        range(len(results)),
        desc=f"strength={strength:g} t={slerp_t:g}",
        leave=False,
    )
    for i in progress:
        row = results.iloc[i]
        true_answers = split_multi_answer(row["Correct Answers"])
        false_answers = split_multi_answer(row["Incorrect Answers"])
        common = dict(
            model=model,
            tokenizer=tokenizer,
            question=row["Question"],
            base_hook_fn=hook,
            layer_name=layer_name,
            device=device,
            use_instruction=not no_instruction,
            model_name=model_name,
        )
        if few_shot:
            scores_true = calculate_scores_fewshot(
                choices=true_answers,
                preset=preset,
                **common,
            )
            scores_false = calculate_scores_fewshot(
                choices=false_answers,
                preset=preset,
                **common,
            )
        else:
            scores_true = [
                calculate_scores_zeroshot(choices=[answer], **common)[0]
                for answer in true_answers
            ]
            scores_false = [
                calculate_scores_zeroshot(choices=[answer], **common)[0]
                for answer in false_answers
            ]
        MC_calcs(
            tag,
            results,
            i,
            scores_true,
            scores_false,
            true_answers,
            format_best(row["Best Answer"]),
        )
        running_mc1 = float(results[f"{tag} MC1"].iloc[: i + 1].mean())
        progress.set_postfix(MC1=f"{running_mc1:.3f}")

    metrics = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    count = max(int(stats.get("prediction_count", 0)), 1)
    summary_stats = {
        "steered_tokens": int(stats.get("steered", 0)),
        "total_tokens": int(stats.get("total", 0)),
        "prediction_count": int(stats.get("prediction_count", 0)),
        "mean_predicted_vector_norm": float(
            stats.get("predicted_norm_sum", 0.0) / count
        ),
    }
    return metrics, summary_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Dyn2D strength x SLERP MC1 ablation")
    parser.add_argument("model_name")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument(
        "--strengths",
        type=_parse_float_list,
        default=DEFAULT_STRENGTHS,
        help='Space/comma separated list, e.g. "0.5 1.0 1.5 2.0 2.5 3.0".',
    )
    parser.add_argument(
        "--slerps",
        type=_parse_float_list,
        default=DEFAULT_SLERPS,
        help='Space/comma separated list, e.g. "0 0.2 0.4 0.6 0.8 1.0".',
    )
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if any(not 0.0 <= value <= 1.0 for value in args.slerps):
        parser.error("all --slerps values must lie in [0, 1]")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint, avg, predictor = load_components(args.checkpoint_path, device)
    fold_idx = int(np.asarray(checkpoint["fold_idx"]).item()) if "fold_idx" in checkpoint.files else None

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device.type == "cpu":
        dtype, device_map = torch.float32, None
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if args.eval_split == "all":
        ids = set(range(len(hf_dataset)))
    else:
        key = "train_q_indices" if args.eval_split == "train" else "test_q_indices"
        if key not in checkpoint.files:
            raise KeyError(f"checkpoint missing {key}")
        ids = set(int(x) for x in np.asarray(checkpoint[key]).reshape(-1))
    base_dataframe = dataframe[dataframe["hf_idx"].isin(ids)].copy().reset_index(drop=True)
    layer_name = get_layer_name(args.model_name, args.layer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dyn2d_ablation_mc1.csv"
    json_path = output_dir / "dyn2d_ablation_results.json"

    cells: list[dict] = []
    outer = tqdm(args.slerps, desc="SLERP rows")
    for slerp_t in outer:
        for strength in args.strengths:
            metrics, steering_stats = evaluate_one(
                model=model,
                tokenizer=tokenizer,
                base_dataframe=base_dataframe,
                model_name=args.model_name,
                layer_name=layer_name,
                device=device,
                avg=avg,
                predictor=predictor,
                strength=float(strength),
                slerp_t=float(slerp_t),
                few_shot=args.few_shot,
                preset=args.preset,
                no_instruction=args.no_instruction,
            )
            cell = {
                "model_name": args.model_name,
                "layer": int(args.layer),
                "fold": fold_idx,
                "strength": float(strength),
                "slerp_t": float(slerp_t),
                **metrics,
                **steering_stats,
            }
            cells.append(cell)
            outer.set_postfix(
                strength=f"{strength:g}",
                t=f"{slerp_t:g}",
                MC1=f"{metrics['MC1']:.4f}",
            )

            # Persist after every cell so long sweeps preserve completed work.
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(cells[0].keys()))
                writer.writeheader()
                writer.writerows(cells)
            report = {
                "experiment": "Dyn2D strength x SLERP ablation",
                "model_name": args.model_name,
                "layer": int(args.layer),
                "fold": fold_idx,
                "checkpoint_path": str(args.checkpoint_path),
                "eval_split": args.eval_split,
                "strengths": [float(x) for x in args.strengths],
                "slerps": [float(x) for x in args.slerps],
                "definition": {
                    "D": "SLERP(normalize(v_avg), normalize(v_hat-v_avg), slerp_t)",
                    "D_scaled": "D * ||v_hat||_2",
                    "intervention": "h_new = h + strength * D_scaled",
                },
                "cells": cells,
            }
            json_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
