"""TruthfulQA generation and LLM-Judge evaluation for L2S-LINEAR."""
from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.S2S_R.eval_utils import HF_NAMES, resolve_device
from methods.S2S_R.evaluate_llm_judge import (
    evaluate_with_judge,
    format_generation_prompt,
    run_judge_phase,
)
from methods.l2s_linear.model import load_l2s_checkpoint
from methods.l2s_linear.steering import get_l2s_linear_intervention


TRUTH_JUDGE_DEFAULT = "models/truthfulqa-truth-judge-llama2-7B"
INFO_JUDGE_DEFAULT = "models/truthfulqa-info-judge-llama2-7B"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def generate_answer(model, tokenizer, question, hook, layer_name, device, *, preset, use_instruction, max_new_tokens, model_name):
    prompt = format_generation_prompt(question, preset, use_instruction, model_name)
    input_device = _model_input_device(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(input_device)
    prompt_len = input_ids.shape[1]
    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            edit = partial(hook, start_idx=generated_ids.shape[1] - 1) if hook is not None else None
            if edit is not None:
                with TraceDict(model, [layer_name], edit_output=edit):
                    outputs = model(generated_ids)
            else:
                outputs = model(generated_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    answer = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()
    if "Q:" in answer:
        answer = answer.split("Q:", 1)[0].strip()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard additive L2S-LINEAR with TruthfulQA LLM Judge")
    parser.add_argument("model_name")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--disable_steering", action="store_true")
    parser.add_argument("--preset", choices=["qa", "null"], default="null")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--truth_judge_path", default=TRUTH_JUDGE_DEFAULT)
    parser.add_argument("--info_judge_path", default=INFO_JUDGE_DEFAULT)
    parser.add_argument("--output_dir", default="./results_llm_judge")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint, mlp = load_l2s_checkpoint(args.checkpoint_path, device)
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

    if "test_q_indices" not in checkpoint.files:
        checkpoint.close()
        raise KeyError("L2S checkpoint missing test_q_indices")
    test_q_indices = sorted(int(x) for x in np.asarray(checkpoint["test_q_indices"]).reshape(-1))
    if args.max_samples is not None:
        test_q_indices = test_q_indices[: args.max_samples]
    layer_name = f"model.layers.{args.layer}"
    stats = {"total": 0, "steered": 0}
    hook = None if args.disable_steering else get_l2s_linear_intervention(
        mlp, alpha=args.alpha, stats=stats
    )

    hf_dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    questions, answers = [], []
    for q_idx in tqdm(test_q_indices, desc="Generating"):
        question = hf_dataset[q_idx]["question"]
        questions.append(question)
        answers.append(
            generate_answer(
                model,
                tokenizer,
                question,
                hook,
                layer_name,
                device,
                preset=args.preset,
                use_instruction=not args.no_instruction,
                max_new_tokens=args.max_new_tokens,
                model_name=args.model_name,
            )
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    truth_scores = run_judge_phase(args.truth_judge_path, questions, answers, info=False, desc="Truth")
    info_scores = run_judge_phase(args.info_judge_path, questions, answers, info=True, desc="Info")
    records = [
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
    fold_idx = int(np.asarray(checkpoint["fold_idx"]).item()) if "fold_idx" in checkpoint.files else 0
    suffix = "baseline" if args.disable_steering else "steered"
    preset_tag = "fewshot" if args.preset == "qa" else "zeroshot"
    output = os.path.join(args.output_dir, f"{args.model_name}_l{args.layer}_fold{fold_idx}_l2s_linear_{suffix}_{preset_tag}.csv")
    pd.DataFrame(records).to_csv(output, index=False)
    avg_truth = float(np.mean(truth_scores)) if truth_scores else float("nan")
    avg_info = float(np.mean(info_scores)) if info_scores else float("nan")
    summary = os.path.join(args.output_dir, "summary.csv")
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as handle:
            handle.write("model,layer,fold,alpha,preset,steering,truth_prob,info_prob,truth_acc,info_acc,steer_rate\n")
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write(
            f"{args.model_name},{args.layer},{fold_idx},{args.alpha},"
            f"{args.preset},{suffix},{avg_truth:.6f},{avg_info:.6f},"
            f"{np.mean([r['truth_acc'] for r in records]):.6f},{np.mean([r['info_acc'] for r in records]):.6f},"
            f"{stats['steered'] / max(stats['total'], 1):.6f}\n"
        )
    checkpoint.close()
    print(f"Saved L2S-LINEAR LLM-Judge results to {output}")


if __name__ == "__main__":
    main()
