"""
Evaluate MC tasks with S2S_R.

Evaluation logic (MC1 style):
  1. For each question, construct full prompts for ALL options.
  2. Calculate log_prob for the option text part only.
  3. Apply length normalization: score = mean(log_probs).
  4. Select option with max score → compare with ground truth.

Supported datasets: COPA, StoryCloze, MMLU, Winogrande, BoolQ

Usage:
    python evaluate_generic.py --model_name llama3.1-8B --dataset copa --layer 14 \
        --checkpoint_path ./checkpoints_generic/copa_S2S_R.npz
"""

import argparse
import json
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from functools import partial
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

# Keep the original dataset and scoring implementation; only the steering
# hook is replaced by the project's S2S_R method.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)
from method.S2S_R.evaluate_predictor import load_components
from method.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
from method.S2S_R.generic.utils_generic import HF_NAMES, MMLU_CATEGORIES, set_seed
from method.S2S_R.generic.prompts_generic import (
    format_boolq_prompt,
    format_copa_prompt,
    format_mmlu_prompt,
    format_storycloze_prompt,
    format_winogrande_prompt,
)


# ============================================================
# Score Calculation
# ============================================================

def calculate_option_score(model, tokenizer, base_prompt, option_text,
                           hook_fn, layer_name, device, normalize_length=True):
    """
    Calculate log probability score for an option.
    Scores only the option tokens (after the prompt).
    """
    prompt_ids = tokenizer(base_prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    intervention_start_idx = prompt_len - 1

    # Append option text (with leading space for raw prompts, as-is for chat)
    if base_prompt.strip().endswith(":"):
        text_to_append = f" {option_text}"
    else:
        text_to_append = option_text

    option_ids = tokenizer(text_to_append, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    option_len = option_ids.shape[1]
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)

    # Hook setup
    current_hook = partial(hook_fn, start_idx=intervention_start_idx) if hook_fn is not None else None

    with torch.no_grad():
        with TraceDict(model, [layer_name], edit_output=current_hook):
            outputs = model(input_ids)

    logits = outputs.logits
    log_probs_all = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1)
    shift_labels = input_ids[..., 1:]
    token_log_probs = torch.gather(log_probs_all, 2, shift_labels.unsqueeze(2)).squeeze(2)

    option_log_probs = token_log_probs[0, intervention_start_idx:intervention_start_idx + option_len]

    if normalize_length:
        return option_log_probs.mean().item()
    else:
        return option_log_probs.sum().item()


# ============================================================
# Per-dataset Evaluation Functions
# ============================================================


def format_progress(correct, total, steering_stats):
    """Format live accuracy, gated steering rate, and weight means."""
    steer_total = steering_stats.get("total", 0)
    steered = steering_stats.get("steered", 0)
    steer_pct = 100.0 * steered / max(steer_total, 1)
    text = f"Acc:{correct / max(total, 1):.4f} Steer:{steer_pct:.1f}%"
    count = steering_stats.get("weight_count", 0)
    if count:
        alpha = steering_stats.get("alpha_sum", 0.0) / count
        beta = steering_stats.get("beta_sum", 0.0) / count
        gamma = steering_stats.get("gamma_sum", 0.0) / count
        text += f" W:{alpha:.3f}/{beta:.3f}/{gamma:.3f}"
    return text

def evaluate_copa(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on COPA validation set (100 samples)."""
    print("Loading COPA validation set...")
    dataset = load_dataset("super_glue", "copa", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        premise = item["premise"]
        q_type = item["question"]
        choice1 = item["choice1"]
        choice2 = item["choice2"]
        correct_idx = item["label"]  # 0 or 1
        choices = [choice1, choice2]

        base_prompt = format_copa_prompt(premise, q_type, choice1, choice2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(format_progress(correct, total, steering_stats))

    return correct / total


def evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on XStoryCloze eval set."""
    print("Loading XStoryCloze (en) eval set...")
    dataset = load_dataset("juletxara/xstory_cloze", "en")['eval']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        story = [item[f'input_sentence_{i}'] for i in range(1, 5)]
        opt1 = item['sentence_quiz1']
        opt2 = item['sentence_quiz2']
        correct_idx = item['answer_right_ending'] - 1  # 1→0, 2→1

        base_prompt = format_storycloze_prompt(story, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(format_progress(correct, total, steering_stats))

    return correct / total


def _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device, items, steering_stats):
    """Shared evaluation loop for MMLU items."""
    correct, total = 0, 0
    pbar = tqdm(items)

    for item in pbar:
        question = item['question']
        choices = item['choices']
        correct_idx = item['answer']

        base_prompt = format_mmlu_prompt(tokenizer, question, choices)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(format_progress(correct, total, steering_stats))

    return correct / total


def evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device,
                         steering_stats, eval_split='test'):
    """
    Evaluate on MMLU Global Balanced (all 4 categories, 500/cat test).
    Returns micro-average accuracy across ~2000 samples.
    """
    print(f"Loading MMLU Global Balanced (split={eval_split})...")

    all_items = []
    for cat_name, subsets in MMLU_CATEGORIES.items():
        loaded = []
        for sub in subsets:
            try:
                local_root=os.environ.get("MMLU_CACHE_DIR", "data/mmlu"); file_split="dev" if eval_split == "train" else eval_split; local_file=os.path.join(local_root, sub, f"{file_split}-00000-of-00001.parquet"); loaded.append(Dataset.from_parquet(local_file))
            except:
                pass
        if not loaded:
            continue

        cat_dataset = concatenate_datasets(loaded).shuffle(seed=42)
        total = len(cat_dataset)

        if eval_split == 'test':
            start_idx, end_idx = 500, min(1000, total)
        else:
            raise ValueError(f"Use eval_split='test' for mmlu_global evaluation")

        if start_idx >= total:
            continue

        dataset = cat_dataset.select(range(start_idx, end_idx))
        print(f"  {cat_name}: {len(dataset)} questions")
        all_items.extend(list(dataset))

    print(f"Total: {len(all_items)} questions")
    return _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device,
                                all_items, steering_stats)


def evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on Winogrande validation set."""
    print("Loading Winogrande validation set...")
    dataset = load_dataset("winogrande", "winogrande_xl")['validation']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        correct_idx = int(item['answer']) - 1  # '1'→0, '2'→1

        base_prompt = format_winogrande_prompt(sentence, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(format_progress(correct, total, steering_stats))

    return correct / total


def evaluate_boolq(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on BoolQ validation set. Options: ["no", "yes"]."""
    print("Loading BoolQ validation set...")
    dataset = load_dataset("super_glue", "boolq", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    options = ["no", "yes"]
    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        passage = item["passage"]
        question = item["question"]
        correct_idx = int(item["label"])  # 0→no, 1→yes

        base_prompt = format_boolq_prompt(passage, question)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in options]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(format_progress(correct, total, steering_stats))

    return correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate MC tasks with S2S_R steering")
    parser.add_argument('--model_name', type=str, default='llama3.1-8B-Instruct',
                        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--weight_kappa', type=float, default=8.0)
    parser.add_argument('--margin', type=float, default=-0.3,
                        help='Only steer when max(beta, gamma) > alpha + margin')
    parser.add_argument('--method', choices=['S2S_R', 'S2S_R_Static'], default='S2S_R')
    parser.add_argument('--disable_steering', action='store_true')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local model directory (overrides model_name)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Optional path for the dataset MC1 result JSON')

    args = parser.parse_args()
    set_seed(args.seed)

    # Load model
    model_path = args.model_dir if args.model_dir else HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype_map[args.dtype], device_map="auto", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the S2S_R checkpoint trained on this dataset's original training split.
    print(f"Loading S2S_R checkpoint: {args.checkpoint_path}")
    _, avg, predictor = load_components(
        args.checkpoint_path, torch.device(device)
    )

    # Setup steering hook. Dataset loading, prompts, option scoring and metrics
    # remain identical to the original generic evaluator.
    layer_name = f"model.layers.{args.layer}"
    steering_stats = {'total': 0, 'steered': 0, 'weight_count': 0}

    if args.disable_steering:
        print(">> Mode: BASELINE (No Steering)")
        hook_fn = None
    else:
        print(
            f">> Mode: S2S_R (kappa={args.weight_kappa}, margin={args.margin})"
        )
        hook_fn = get_s2s_r_intervention(
            avg,
            predictor,
            weight_kappa=args.weight_kappa,
            margin=args.margin,
            method=args.method,
            stats=steering_stats,
        )

    # Dispatch evaluation
    if args.dataset == 'copa':
        accuracy = evaluate_copa(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'storycloze':
        accuracy = evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'mmlu_global':
        accuracy = evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'winogrande':
        accuracy = evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'boolq':
        accuracy = evaluate_boolq(model, tokenizer, hook_fn, layer_name, device, steering_stats)

    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_name}")
    print(f"Layer:      {args.layer}")
    print(f"Steering:   {'OFF' if args.disable_steering else 'ON'}")
    if not args.disable_steering:
        print(
            f"Parameters: kappa={args.weight_kappa}, margin={args.margin}, "
        )
        if steering_stats['total'] > 0:
            steer_pct = steering_stats['steered'] / steering_stats['total'] * 100
            print(f"Stats:      Steered={steering_stats['steered']}/{steering_stats['total']} ({steer_pct:.1f}%)")
        if steering_stats.get("weight_count", 0):
            count = steering_stats["weight_count"]
            print(
                "Mean weights (alpha/beta/gamma): "
                f"{steering_stats['alpha_sum'] / count:.6f} / "
                f"{steering_stats['beta_sum'] / count:.6f} / "
                f"{steering_stats['gamma_sum'] / count:.6f}"
            )
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print("=" * 50)

    if args.output_path:
        count = steering_stats.get("weight_count", 0)
        result = {
            "dataset": args.dataset,
            "metric": "MC1",
            "MC1": float(accuracy),
            "model_name": args.model_name,
            "layer": int(args.layer),
            "checkpoint_path": args.checkpoint_path,
            "steering_enabled": not args.disable_steering,
            "parameters": {
                "weight_kappa": args.weight_kappa,
                "margin": args.margin,
                "seed": args.seed,
            },
            "steering_stats": {
                "steered_tokens": int(steering_stats.get("steered", 0)),
                "total_tokens": int(steering_stats.get("total", 0)),
                "steer_token_ratio": (
                    steering_stats.get("steered", 0)
                    / max(steering_stats.get("total", 0), 1)
                ),
                "mean_adaptive_weights": (
                    {
                        "alpha": steering_stats["alpha_sum"] / count,
                        "beta": steering_stats["beta_sum"] / count,
                        "gamma": steering_stats["gamma_sum"] / count,
                    }
                    if count else None
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
