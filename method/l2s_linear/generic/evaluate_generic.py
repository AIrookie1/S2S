"""Evaluate generic multiple-choice tasks with additive L2S-LINEAR steering.

This module intentionally reuses the canonical generic dataset loading,
prompt construction, option scoring, and MC1 evaluation functions used by
Dyn3D. Only the checkpoint loader and steering hook are L2S-LINEAR-specific.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.dyn3d_sga.generic.evaluate_generic import (
    evaluate_boolq,
    evaluate_copa,
    evaluate_mmlu_global,
    evaluate_storycloze,
    evaluate_winogrande,
)
from method.dyn3d_sga.generic.utils_generic import HF_NAMES, set_seed
from method.l2s_linear.model import load_l2s_checkpoint
from method.l2s_linear.steering import get_l2s_linear_intervention

DATASETS = ("copa", "storycloze", "mmlu_global", "winogrande", "boolq")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generic MC tasks with additive L2S-LINEAR steering"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8B-Instruct",
        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}",
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Additive steering strength in h_new = h + alpha * L2S(h)",
    )
    parser.add_argument("--disable_steering", action="store_true")
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Local model directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path for the dataset MC1 result JSON",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cpu":
        torch_dtype = torch.float32
        device_map = None
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    print(f"Loading L2S-LINEAR checkpoint: {args.checkpoint_path}")
    checkpoint, mlp = load_l2s_checkpoint(args.checkpoint_path, device)

    layer_name = f"model.layers.{args.layer}"
    steering_stats = {"total": 0, "steered": 0}
    if args.disable_steering:
        print(">> Mode: BASELINE (No Steering)")
        hook_fn = None
    else:
        print(f">> Mode: L2S-LINEAR (alpha={args.alpha})")
        hook_fn = get_l2s_linear_intervention(
            mlp,
            alpha=args.alpha,
            stats=steering_stats,
        )

    evaluators = {
        "copa": evaluate_copa,
        "storycloze": evaluate_storycloze,
        "mmlu_global": evaluate_mmlu_global,
        "winogrande": evaluate_winogrande,
        "boolq": evaluate_boolq,
    }
    try:
        accuracy = evaluators[args.dataset](
            model,
            tokenizer,
            hook_fn,
            layer_name,
            device,
            steering_stats,
        )
    finally:
        checkpoint.close()

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_name}")
    print(f"Layer:      {args.layer}")
    print(f"Steering:   {'OFF' if args.disable_steering else 'ON'}")
    if not args.disable_steering:
        print(f"Parameters: alpha={args.alpha}")
        if steering_stats["total"] > 0:
            steer_pct = steering_stats["steered"] / steering_stats["total"] * 100.0
            print(
                "Stats:      "
                f"Steered={steering_stats['steered']}/{steering_stats['total']} "
                f"({steer_pct:.1f}%)"
            )
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print("=" * 50)

    if args.output_path:
        result = {
            "method": "L2S-LINEAR",
            "dataset": args.dataset,
            "metric": "MC1",
            "MC1": float(accuracy),
            "model_name": args.model_name,
            "layer": int(args.layer),
            "checkpoint_path": args.checkpoint_path,
            "steering_enabled": not args.disable_steering,
            "parameters": {
                "alpha": float(args.alpha),
                "seed": int(args.seed),
            },
            "steering_stats": {
                "steered_tokens": int(steering_stats.get("steered", 0)),
                "total_tokens": int(steering_stats.get("total", 0)),
                "steer_token_ratio": (
                    steering_stats.get("steered", 0)
                    / max(steering_stats.get("total", 0), 1)
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
