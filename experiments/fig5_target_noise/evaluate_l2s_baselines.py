#!/usr/bin/env python3
"""TruthfulQA MC comparison: L2S-LINEAR vs the current L2S-SLERP rule."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.S2S_R.eval_utils import (
    HF_NAMES, calculate_scores_fewshot, calculate_scores_zeroshot, format_best,
    get_layer_name, load_truthfulqa_mc_dataframe, resolve_device, set_seed,
    split_multi_answer,
)
from methods.l2s_linear.model import load_l2s_checkpoint as load_linear_checkpoint
from methods.l2s_linear.steering import get_l2s_linear_intervention
from methods.l2s_slerp.model import load_l2s_checkpoint as load_slerp_checkpoint
from methods.l2s_slerp.steering import get_l2s_slerp_intervention
from truthfulqa.models import MC_calcs, set_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare L2S-LINEAR and L2S-SLERP on TruthfulQA MC")
    p.add_argument("model_name")
    p.add_argument("--linear_checkpoint_path", required=True)
    p.add_argument("--slerp_checkpoint_path", required=True)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--csv_path", default="./data/truthfulqa/TruthfulQA.csv")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--linear_alpha", type=float, default=1.4)
    p.add_argument("--slerp_kappa", type=float, default=20.0)
    p.add_argument("--slerp_alpha", type=float, default=0.6)
    p.add_argument("--slerp_beta", type=float, default=0.4)
    p.add_argument("--few_shot", action="store_true")
    p.add_argument("--preset", default="qa")
    p.add_argument("--no_instruction", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--residual_noise_sigma", type=float, default=0.0,
                   help="Inference-only relative Gaussian noise on predicted residual direction.")
    p.add_argument("--residual_noise_seed", type=int, default=42,
                   help="Seed for inference-only residual noise.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--output_dir", default="./results/experiment/truthfulqa_l2s_linear_vs_l2s_slerp")
    p.add_argument("--output_stem", default="l2s_linear_vs_l2s_slerp")
    p.add_argument("--save_details", action="store_true")
    args = p.parse_args()
    for name in ("linear_alpha", "slerp_alpha", "slerp_beta"):
        if not math.isfinite(getattr(args, name)):
            p.error(f"--{name} must be finite")
    if not math.isfinite(args.slerp_kappa) or args.slerp_kappa <= 0:
        p.error("--slerp_kappa must be finite and positive")
    if args.slerp_beta >= 1.0:
        p.error("--slerp_beta must be smaller than 1")
    return args


def checkpoint_split(checkpoint, eval_split: str):
    if eval_split == "all":
        return None
    key = "train_q_indices" if eval_split == "train" else "test_q_indices"
    if key not in checkpoint.files:
        raise KeyError(f"checkpoint missing {key}")
    return set(int(x) for x in np.asarray(checkpoint[key]).reshape(-1))


def evaluate_condition(*, label: str, base_dataframe, model, tokenizer, hook, layer_name: str,
                       device: torch.device, model_name: str, few_shot: bool, preset: str,
                       no_instruction: bool, stats: dict, save_details_path: Path | None) -> dict[str, Any]:
    dataframe = base_dataframe.copy().reset_index(drop=True)
    tag = model_name
    set_columns(tag, dataframe)
    mode = "fewshot" if few_shot else "zeroshot"
    progress = tqdm(range(len(dataframe)), desc=label)
    for i in progress:
        row = dataframe.iloc[i]
        true = split_multi_answer(row["Correct Answers"])
        false = split_multi_answer(row["Incorrect Answers"])
        common = dict(model=model, tokenizer=tokenizer, question=row["Question"],
                      base_hook_fn=hook, layer_name=layer_name, device=device,
                      use_instruction=not no_instruction, model_name=model_name)
        if few_shot:
            scores_true = calculate_scores_fewshot(choices=true, preset=preset, **common)
            scores_false = calculate_scores_fewshot(choices=false, preset=preset, **common)
        else:
            scores_true = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in true]
            scores_false = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in false]
        MC_calcs(tag, dataframe, i, scores_true, scores_false, true, format_best(row["Best Answer"]))
        running = {m: float(dataframe[f"{tag} {m}"].iloc[: i + 1].mean()) for m in ("MC1", "MC2", "MC3")}
        progress.set_description(
            f"{label} MC1:{running['MC1']:.3f} MC2:{running['MC2']:.3f} "
            f"MC3:{running['MC3']:.3f} steer:{stats.get('steered',0)/max(stats.get('total',0),1):.3f}"
        )
    metrics = {m: float(dataframe[f"{tag} {m}"].mean()) for m in ("MC1", "MC2", "MC3")}
    metrics.update({
        "steering_rate": stats.get("steered", 0) / max(stats.get("total", 0), 1),
        "steered_tokens": int(stats.get("steered", 0)),
        "total_tokens": int(stats.get("total", 0)),
        "questions": int(len(dataframe)),
        "mode": mode,
    })
    if save_details_path is not None:
        save_details_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(save_details_path, index=False)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    linear_cp, linear_mlp = load_linear_checkpoint(args.linear_checkpoint_path, device)
    slerp_cp, slerp_mlp = load_slerp_checkpoint(args.slerp_checkpoint_path, device)
    try:
        linear_q = checkpoint_split(linear_cp, args.eval_split)
        slerp_q = checkpoint_split(slerp_cp, args.eval_split)
        if "v_avg" not in linear_cp.files or "v_avg" not in slerp_cp.files:
            raise KeyError(
                "L2S checkpoint missing v_avg. Retrain the clean checkpoint with the "
                "updated train_l2s.py before running inference residual noise."
            )
        linear_avg = torch.as_tensor(np.asarray(linear_cp["v_avg"]), dtype=torch.float32, device=device)
        slerp_avg = torch.as_tensor(np.asarray(slerp_cp["v_avg"]), dtype=torch.float32, device=device)
        if linear_q != slerp_q:
            raise ValueError("L2S-LINEAR and L2S-SLERP checkpoints use different evaluation splits")
        fold_linear = int(np.asarray(linear_cp["fold_idx"]).item()) if "fold_idx" in linear_cp.files else -1
        fold_slerp = int(np.asarray(slerp_cp["fold_idx"]).item()) if "fold_idx" in slerp_cp.files else -1
        if fold_linear != fold_slerp:
            raise ValueError("L2S-LINEAR and L2S-SLERP checkpoints use different folds")

        model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device.type == "cpu":
            dtype, device_map = torch.float32, None
        else:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
            device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        if device.type == "cpu":
            model = model.to(device)
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        dataframe, _ = load_truthfulqa_mc_dataframe(args.csv_path)
        if linear_q is not None:
            dataframe = dataframe[dataframe["hf_idx"].isin(linear_q)].copy().reset_index(drop=True)
        layer_name = get_layer_name(args.model_name, args.layer)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        linear_stats = {"total": 0, "steered": 0}
        linear_metrics = evaluate_condition(
            label="L2S-LINEAR", base_dataframe=dataframe, model=model, tokenizer=tokenizer,
            hook=get_l2s_linear_intervention(
                linear_mlp, alpha=args.linear_alpha, stats=linear_stats,
                v_avg=linear_avg, residual_noise_sigma=args.residual_noise_sigma,
                residual_noise_seed=args.residual_noise_seed,
            ),
            layer_name=layer_name, device=device, model_name=args.model_name, few_shot=args.few_shot,
            preset=args.preset, no_instruction=args.no_instruction, stats=linear_stats,
            save_details_path=(output_dir / f"{args.output_stem}_l2s_linear_details.csv") if args.save_details else None,
        )
        results.append({"method": "l2s_linear", "alpha": float(args.linear_alpha), "kappa": None, "beta": None, **linear_metrics})

        slerp_stats = {"total": 0, "steered": 0}
        slerp_metrics = evaluate_condition(
            label="L2S-SLERP", base_dataframe=dataframe, model=model, tokenizer=tokenizer,
            hook=get_l2s_slerp_intervention(
                slerp_mlp, kappa=args.slerp_kappa, alpha=args.slerp_alpha,
                beta=args.slerp_beta, stats=slerp_stats, v_avg=slerp_avg,
                residual_noise_sigma=args.residual_noise_sigma,
                residual_noise_seed=args.residual_noise_seed,
            ),
            layer_name=layer_name, device=device, model_name=args.model_name, few_shot=args.few_shot,
            preset=args.preset, no_instruction=args.no_instruction, stats=slerp_stats,
            save_details_path=(output_dir / f"{args.output_stem}_l2s_slerp_details.csv") if args.save_details else None,
        )
        results.append({"method": "l2s_slerp", "alpha": float(args.slerp_alpha), "kappa": float(args.slerp_kappa), "beta": float(args.slerp_beta), **slerp_metrics})

        payload = {
            "experiment": "truthfulqa_l2s_linear_vs_l2s_slerp",
            "model_name": args.model_name, "model_path": str(model_path), "layer": int(args.layer),
            "fold_idx": fold_linear, "eval_split": args.eval_split,
            "residual_noise_sigma": float(args.residual_noise_sigma),
            "residual_noise_seed": int(args.residual_noise_seed),
            "linear_checkpoint_path": str(args.linear_checkpoint_path),
            "slerp_checkpoint_path": str(args.slerp_checkpoint_path),
            "definitions": {
                "residual_noise": "r_noisy = normalize(r + sigma*||r||*normalize(epsilon)); epsilon~N(0,I); inference only",
                "l2s_residual": "r = normalize(L2S(h)-v_avg); preserve ||L2S(h)-v_avg|| when reconstructing noisy v_hat",
                "l2s_linear": "h_new = h + alpha * noisy_L2S(h)",
                "l2s_slerp": "vMF-gated spherical rotation toward normalize(noisy_L2S(h)), preserving ||h||",
            },
            "results": results,
        }
        json_path = output_dir / f"{args.output_stem}.json"
        csv_path = output_dir / f"{args.output_stem}.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fields = ["method", "alpha", "kappa", "beta", "MC1", "MC2", "MC3", "steering_rate", "steered_tokens", "total_tokens", "questions", "mode"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(results)
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV : {csv_path}")
    finally:
        linear_cp.close(); slerp_cp.close()


if __name__ == "__main__":
    main()
