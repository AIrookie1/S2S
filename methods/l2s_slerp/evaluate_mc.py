"""TruthfulQA multiple-choice evaluation for L2S-SLERP."""
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

from methods.S2S_R.eval_utils import (
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
from methods.l2s_slerp.model import load_l2s_checkpoint
from methods.l2s_slerp.steering import get_l2s_slerp_intervention
from truthfulqa.models import MC_calcs, set_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate L2S-SLERP on TruthfulQA MC")
    parser.add_argument("model_name")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./data/truthfulqa/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint, mlp = load_l2s_checkpoint(args.checkpoint_path, device)
    if args.eval_split == "all":
        eval_q_indices = None
    else:
        key = "train_q_indices" if args.eval_split == "train" else "test_q_indices"
        if key not in checkpoint.files:
            checkpoint.close()
            raise KeyError(f"L2S checkpoint missing {key}")
        eval_q_indices = set(int(x) for x in np.asarray(checkpoint[key]).reshape(-1))

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
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if eval_q_indices is not None:
        dataframe = dataframe[dataframe["hf_idx"].isin(eval_q_indices)].copy().reset_index(drop=True)
    tag = args.model_name
    set_columns(tag, dataframe)
    layer_name = get_layer_name(args.model_name, args.layer)
    mode = "fewshot" if args.few_shot else "zeroshot"
    if args.output_path is None:
        args.output_path = (
            f"results/{args.model_name.replace('/', '_')}_l{args.layer}_fold"
            f"{int(np.asarray(checkpoint['fold_idx']).item()) if 'fold_idx' in checkpoint.files else 'unknown'}_l2s_slerp_{mode}.json"
        )

    stats = {"total": 0, "steered": 0}
    hook = None
    if not args.no_intervention:
        hook = get_l2s_slerp_intervention(
            mlp,
            kappa=args.kappa,
            alpha=args.alpha,
            beta=args.beta,
            stats=stats,
        )
    print(
        f"L2S-SLERP | dim={mlp.input_dim} hidden={mlp.hidden_dim} "
        f"kappa={args.kappa} alpha={args.alpha} beta={args.beta} "
        f"split={args.eval_split} questions={len(dataframe)}"
    )

    progress = tqdm(range(len(dataframe)), desc=mode)
    for i in progress:
        row = dataframe.iloc[i]
        true = split_multi_answer(row["Correct Answers"])
        false = split_multi_answer(row["Incorrect Answers"])
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
            scores_true = calculate_scores_fewshot(choices=true, preset=args.preset, **common)
            scores_false = calculate_scores_fewshot(choices=false, preset=args.preset, **common)
        else:
            scores_true = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in true]
            scores_false = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in false]
        MC_calcs(tag, dataframe, i, scores_true, scores_false, true, format_best(row["Best Answer"]))
        running = {
            metric: float(dataframe[f"{tag} {metric}"].iloc[: i + 1].mean())
            for metric in ("MC1", "MC2", "MC3")
        }
        progress.set_description(
            f"MC1:{running['MC1']:.3f} MC2:{running['MC2']:.3f} "
            f"MC3:{running['MC3']:.3f} steer:{stats['steered'] / max(stats['total'], 1):.3f}"
        )

    metrics = {
        metric: float(dataframe[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    metrics.update(
        {
            "steering_rate": stats["steered"] / max(stats["total"], 1),
            "steered_tokens": int(stats["steered"]),
            "total_tokens": int(stats["total"]),
        }
    )
    report = {
        "method": "L2S-SLERP",
        "model_name": args.model_name,
        "layer": int(args.layer),
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "parameters": {
            "kappa": args.kappa,
            "alpha": args.alpha,
            "beta": args.beta,
            "few_shot": bool(args.few_shot),
            "preset": args.preset,
            "no_intervention": bool(args.no_intervention),
        },
        "metrics": metrics,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.save_details:
        dataframe.to_csv(output.with_name(output.stem + "_details.csv"), index=False)
    checkpoint.close()
    print(f"\nMC1: {metrics['MC1']:.4f} | MC2: {metrics['MC2']:.4f} | MC3: {metrics['MC3']:.4f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
