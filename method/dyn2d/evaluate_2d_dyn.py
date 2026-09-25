"""TruthfulQA MC evaluation for Dyn2D steering."""
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dyn2D: SLERP(normalize(v_avg), "
            "normalize(v_hat-v_avg), t), restore ||v_hat||, then additive steering"
        )
    )
    parser.add_argument("model_name")
    parser.add_argument(
        "--checkpoint_path",
        "--operator_path",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument(
        "--strength",
        "--alpha",
        dest="strength",
        type=float,
        default=1.0,
        help="Standard additive steering strength in h_new = h + strength * D_scaled.",
    )
    parser.add_argument(
        "--slerp_t",
        "--slerp",
        dest="slerp_t",
        type=float,
        default=0.5,
        help="SLERP coefficient: 0 -> normalize(v_avg), 1 -> normalize(v_hat-v_avg).",
    )
    parser.add_argument("--method", choices=["S2S_L", "S2S_L_Static"], default="S2S_L_Static")
    parser.add_argument("--weight_kappa", type=float, default=3.0)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--slerp_eps", type=float, default=1e-8)
    parser.add_argument("--small_angle_threshold", type=float, default=1e-4)
    parser.add_argument(
        "--residual_noise_sigma", type=float, default=0.0,
        help="Inference-only relative Gaussian noise on the predicted residual direction.",
    )
    parser.add_argument(
        "--residual_noise_seed", type=int, default=42,
        help="Seed for inference-only residual noise.",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.strength):
        parser.error("--strength must be finite")
    if not np.isfinite(args.slerp_t) or not 0.0 <= args.slerp_t <= 1.0:
        parser.error("--slerp_t must be in [0, 1]")
    if not np.isfinite(args.residual_noise_sigma) or args.residual_noise_sigma < 0:
        parser.error("--residual_noise_sigma must be finite and non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint, avg, predictor = load_components(args.checkpoint_path, device)
    print(
        f"Dyn2D | dim={avg.numel()} mlp_hidden={predictor.hidden_dim} "
        f"strength={args.strength} slerp_t={args.slerp_t}"
    )

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

    results = dataframe[dataframe["hf_idx"].isin(ids)].copy().reset_index(drop=True)
    tag = args.model_name
    set_columns(tag, results)
    mode = "few_shot" if args.few_shot else "zero_shot"
    layer_name = get_layer_name(args.model_name, args.layer)

    fold_idx = None
    if "fold_idx" in checkpoint.files:
        fold_idx = int(np.asarray(checkpoint["fold_idx"]).item())

    if args.output_path is None:
        fold_text = f"_fold{fold_idx}" if fold_idx is not None else ""
        args.output_path = (
            f"results/{args.model_name.replace('/', '_')}_l{args.layer}{fold_text}"
            f"_dyn2d_strength{args.strength:g}_slerp{args.slerp_t:g}_{mode}.json"
        )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "steered": 0, "prediction_count": 0}
    hook = None if args.no_intervention else get_2d_dynamic_intervention(
        avg,
        predictor,
        strength=args.strength,
        slerp_t=args.slerp_t,
        eps=args.slerp_eps,
        small_angle_threshold=args.small_angle_threshold,
        method=args.method,
        weight_kappa=args.weight_kappa,
        residual_noise_sigma=args.residual_noise_sigma,
        residual_noise_seed=args.residual_noise_seed,
        stats=stats,
    )

    progress = tqdm(range(len(results)), desc=mode)
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
            use_instruction=not args.no_instruction,
            model_name=args.model_name,
        )
        if args.few_shot:
            scores_true = calculate_scores_fewshot(
                choices=true_answers,
                preset=args.preset,
                **common,
            )
            scores_false = calculate_scores_fewshot(
                choices=false_answers,
                preset=args.preset,
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
        running = {
            metric: float(results[f"{tag} {metric}"].iloc[: i + 1].mean())
            for metric in ("MC1", "MC2", "MC3")
        }
        progress.set_description(
            f"MC1:{running['MC1']:.3f} MC2:{running['MC2']:.3f} "
            f"MC3:{running['MC3']:.3f}"
        )
        if args.save_details:
            results.loc[i, f"{tag} scores_true"] = str(scores_true)
            results.loc[i, f"{tag} scores_false"] = str(scores_false)

    metric_values = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    prediction_count = max(int(stats.get("prediction_count", 0)), 1)
    steering_stats = {
        "steered_tokens": int(stats.get("steered", 0)),
        "total_tokens": int(stats.get("total", 0)),
        "steer_token_ratio": (
            stats.get("steered", 0) / max(stats.get("total", 0), 1)
        ),
        "prediction_count": int(stats.get("prediction_count", 0)),
        "mean_predicted_vector_norm": (
            stats.get("predicted_norm_sum", 0.0) / prediction_count
        ),
        "mean_applied_steering_norm": (
            stats.get("scaled_norm_sum", 0.0) / prediction_count
        ),
    }
    report = {
        "method": "Dyn2D SLERP + predicted-norm additive steering",
        "definition": {
            "V": "normalize(v_avg)",
            "R": "normalize(v_hat - v_avg)",
            "D": "SLERP(V, R, slerp_t)",
            "D_scaled": "D * ||v_hat||_2",
            "intervention": "h_new = h + strength * D_scaled",
        },
        "model_name": args.model_name,
        "layer": int(args.layer),
        "fold_idx": fold_idx,
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "parameters": {
            "strength": float(args.strength),
            "slerp_t": float(args.slerp_t),
            "few_shot": bool(args.few_shot),
            "preset": args.preset,
            "steering_enabled": not args.no_intervention,
            "residual_noise_sigma": float(args.residual_noise_sigma),
            "residual_noise_seed": int(args.residual_noise_seed),
        },
        "metrics": metric_values,
        "steering_stats": steering_stats,
        "results": json.loads(results.to_json(orient="records", force_ascii=False)),
    }
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved results to {args.output_path}")
    print(
        f"MC1: {metric_values['MC1']:.4f} | MC2: {metric_values['MC2']:.4f} | "
        f"MC3: {metric_values['MC3']:.4f}"
    )


if __name__ == "__main__":
    main()
