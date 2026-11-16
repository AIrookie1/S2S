"""Evaluate generic multiple-choice tasks with S2S_L steering."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.S2S_L.steering import get_s2s_l_intervention
from method.S2S_R.evaluate_predictor import load_components
from method.S2S_R.generic.evaluate_generic import (
    evaluate_boolq,
    evaluate_copa,
    evaluate_mmlu_global,
    evaluate_storycloze,
    evaluate_winogrande,
)
from method.S2S_R.generic.utils_generic import HF_NAMES, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MC tasks with S2S_L steering")
    parser.add_argument("--model_name", type=str, default="llama3.1-8B-Instruct")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["copa", "storycloze", "mmlu_global", "winogrande", "boolq"],
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--strength", "--alpha", dest="strength", type=float, default=1.0)
    parser.add_argument("--slerp_t", "--slerp", dest="slerp_t", type=float, default=0.5)
    parser.add_argument("--method", choices=["S2S_L", "S2S_L_Static"], default="S2S_L_Static")
    parser.add_argument("--weight_kappa", type=float, default=3.0)
    parser.add_argument("--disable_steering", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if not np.isfinite(args.strength):
        parser.error("--strength must be finite")
    if not np.isfinite(args.slerp_t) or not 0.0 <= args.slerp_t <= 1.0:
        parser.error("--slerp_t must be in [0, 1]")

    set_seed(args.seed)
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading direct-MLP checkpoint: {args.checkpoint_path}")
    _, avg, predictor = load_components(args.checkpoint_path, torch.device(device))

    layer_name = f"model.layers.{args.layer}"
    steering_stats = {"total": 0, "steered": 0, "prediction_count": 0}
    if args.disable_steering:
        print(">> Mode: BASELINE (No Steering)")
        hook_fn = None
    else:
        print(
            f">> Mode: S2S_L (strength={args.strength}, slerp_t={args.slerp_t})"
        )
        hook_fn = get_s2s_l_intervention(
            avg,
            predictor,
            strength=args.strength,
            slerp_t=args.slerp_t,
            method=args.method,
            weight_kappa=args.weight_kappa,
            stats=steering_stats,
        )

    evaluators = {
        "copa": evaluate_copa,
        "storycloze": evaluate_storycloze,
        "mmlu_global": evaluate_mmlu_global,
        "winogrande": evaluate_winogrande,
        "boolq": evaluate_boolq,
    }
    accuracy = evaluators[args.dataset](
        model,
        tokenizer,
        hook_fn,
        layer_name,
        device,
        steering_stats,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_name}")
    print(f"Layer:      {args.layer}")
    print(f"Steering:   {'OFF' if args.disable_steering else 'ON'}")
    if not args.disable_steering:
        print(f"Parameters: strength={args.strength}, slerp_t={args.slerp_t}")
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print("=" * 50)

    if args.output_path:
        prediction_count = max(int(steering_stats.get("prediction_count", 0)), 1)
        result = {
            "method": "S2S_L",
            "dataset": args.dataset,
            "metric": "MC1",
            "MC1": float(accuracy),
            "model_name": args.model_name,
            "layer": int(args.layer),
            "checkpoint_path": args.checkpoint_path,
            "steering_enabled": not args.disable_steering,
            "parameters": {
                "strength": float(args.strength),
                "slerp_t": float(args.slerp_t),
                "seed": int(args.seed),
            },
            "steering_stats": {
                "steered_tokens": int(steering_stats.get("steered", 0)),
                "total_tokens": int(steering_stats.get("total", 0)),
                "steer_token_ratio": (
                    steering_stats.get("steered", 0)
                    / max(steering_stats.get("total", 0), 1)
                ),
                "mean_predicted_vector_norm": (
                    steering_stats.get("predicted_norm_sum", 0.0) / prediction_count
                ),
            },
        }
        output_path = os.path.abspath(args.output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved MC1 result to {output_path}")


if __name__ == "__main__":
    main()
