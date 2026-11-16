"""TruthfulQA Truth/Info judge evaluation with S2S_L steering."""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.S2S_L.steering import get_s2s_l_intervention
from method.S2S_R.evaluate_predictor import load_components
from method.S2S_R.evaluate_llm_judge import (
    HF_NAMES,
    INFO_JUDGE_DEFAULT,
    TRUTH_JUDGE_DEFAULT,
    generate_answer,
    run_judge_phase,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Judge evaluation with S2S_L steering")
    parser.add_argument("model_name", type=str)
    parser.add_argument("--checkpoint_path", "--prototype_path", dest="checkpoint_path", required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--strength", "--alpha", dest="strength", type=float, default=1.0)
    parser.add_argument("--slerp_t", "--slerp", dest="slerp_t", type=float, default=0.5)
    parser.add_argument("--disable_steering", action="store_true")
    parser.add_argument("--preset", type=str, default="null", choices=["qa", "null"])
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--truth_judge_path", type=str, default=TRUTH_JUDGE_DEFAULT)
    parser.add_argument("--info_judge_path", type=str, default=INFO_JUDGE_DEFAULT)
    parser.add_argument("--output_dir", type=str, default="./results_llm_judge")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not np.isfinite(args.strength):
        parser.error("--strength must be finite")
    if not np.isfinite(args.slerp_t) or not 0.0 <= args.slerp_t <= 1.0:
        parser.error("--slerp_t must be in [0, 1]")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading direct-MLP checkpoint: {args.checkpoint_path}")
    checkpoint, avg, predictor = load_components(args.checkpoint_path, torch.device(device))
    test_q_indices = set(int(x) for x in np.asarray(checkpoint["test_q_indices"]).reshape(-1))
    fold_match = re.search(r"fold(\d+)", args.checkpoint_path)
    fold_idx = int(fold_match.group(1)) if fold_match else int(
        np.asarray(checkpoint["fold_idx"]).item() if "fold_idx" in checkpoint.files else 0
    )

    layer_name = f"model.layers.{args.layer}"
    steering_stats = {"total": 0, "steered": 0, "prediction_count": 0}
    if args.disable_steering:
        base_hook_fn = None
        print("Steering: OFF (Baseline)")
    else:
        base_hook_fn = get_s2s_l_intervention(
            avg,
            predictor,
            strength=args.strength,
            slerp_t=args.slerp_t,
            stats=steering_stats,
        )
        print(f"Steering: ON (strength={args.strength}, slerp_t={args.slerp_t})")

    hf_dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    test_list = sorted(test_q_indices)
    if args.max_samples:
        test_list = test_list[: args.max_samples]

    questions, answers = [], []
    for q_idx in tqdm(test_list, desc="Generating"):
        question = hf_dataset[q_idx]["question"]
        questions.append(question)
        answers.append(
            generate_answer(
                model,
                tokenizer,
                question,
                base_hook_fn,
                layer_name,
                device,
                preset=args.preset,
                use_instruction=not args.no_instruction,
                max_new_tokens=args.max_new_tokens,
                model_name=args.model_name,
            )
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    truth_scores = run_judge_phase(
        args.truth_judge_path, questions, answers, info=False, desc="Truth"
    )
    info_scores = run_judge_phase(
        args.info_judge_path, questions, answers, info=True, desc="Info"
    )

    rows = [
        {
            "question": questions[i],
            "answer": answers[i],
            "truth_prob": truth_scores[i],
            "info_prob": info_scores[i],
            "truth_acc": int(truth_scores[i] >= 0.5),
            "info_acc": int(info_scores[i] >= 0.5),
        }
        for i in range(len(questions))
    ]
    avg_truth = float(np.mean(truth_scores)) if truth_scores else 0.0
    avg_info = float(np.mean(info_scores)) if info_scores else 0.0
    truth_acc = float(np.mean([row["truth_acc"] for row in rows])) if rows else 0.0
    info_acc = float(np.mean([row["info_acc"] for row in rows])) if rows else 0.0

    if args.disable_steering:
        suffix = "baseline"
    else:
        suffix = f"S2S_L_strength{args.strength:g}_slerp{args.slerp_t:g}"
    preset_suffix = "fewshot" if args.preset == "qa" else "zeroshot"
    out_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_l{args.layer}_fold{fold_idx}_{suffix}_{preset_suffix}.csv",
    )
    pd.DataFrame(rows).to_csv(out_file, index=False)

    print(f"Truth: {avg_truth:.4f} | Info: {avg_info:.4f} | T*I: {avg_truth * avg_info:.4f}")
    print(f"Truth Acc: {truth_acc:.4f} | Info Acc: {info_acc:.4f}")
    print(f"Saved to: {out_file}")

    summary = os.path.join(args.output_dir, "summary.csv")
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as handle:
            handle.write(
                "model,layer,fold,strength,slerp_t,preset,steering,truth_prob,info_prob,true_x_info,truth_acc,info_acc\n"
            )
    with open(summary, "a", encoding="utf-8") as handle:
        steer = "off" if args.disable_steering else "on"
        strength = 0.0 if args.disable_steering else args.strength
        handle.write(
            f"{args.model_name},{args.layer},{fold_idx},{strength},{args.slerp_t},"
            f"{args.preset},{steer},{avg_truth:.4f},{avg_info:.4f},"
            f"{avg_truth * avg_info:.4f},{truth_acc:.4f},{info_acc:.4f}\n"
        )


if __name__ == "__main__":
    main()
