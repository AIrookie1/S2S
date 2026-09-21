#!/usr/bin/env python3
"""Three-pass TruthfulQA LLM-judge evaluation for fixed-predictor Dyn2D.

Pass 1/2 reuse the same cache protocol as Dyn3D. Pass 3 freezes the direct-MLP
output computed from cached Q+A0 last-token hidden state, then applies Dyn2D at
every autoregressive step:

    D = SLERP(normalize(v_avg), normalize(v_hat-v_avg), slerp_t)
    h_new = h + strength * D * ||v_hat||_2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn2d.steering import get_2d_dynamic_intervention
from method.dyn3d_sga.evaluate_dyn_mlp import load_components
from method.dyn3d_sga.evaluate_llm_judge import format_generation_prompt
from method.dyn3d_sga.evaluate_llm_judge_3d import (
    _atomic_torch_save,
    cache_record_is_valid,
    capture_probe_last_hidden,
    controls_from_cached_hidden,
    generate_answer,
    load_or_init_cache,
    run_judge_phase,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-pass fixed-predictor Dyn2D TruthfulQA LLM-judge evaluation"
    )
    parser.add_argument("model_name")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--truth_judge_path", required=True)
    parser.add_argument("--info_judge_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--preset", type=str, default="null", choices=["qa", "null"])
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--strength", "--alpha", dest="strength", type=float, default=1.0)
    parser.add_argument("--slerp_t", "--slerp", dest="slerp_t", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--disable_steering", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.strength):
        parser.error("--strength must be finite")
    if not np.isfinite(args.slerp_t) or not 0.0 <= args.slerp_t <= 1.0:
        parser.error("--slerp_t must be in [0, 1]")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_path is None:
        args.cache_path = str(
            Path(args.output_dir) / f"{args.model_name}_l{args.layer}_base_probe_cache.pt"
        )
    if args.output_path is None:
        args.output_path = str(
            Path(args.output_dir)
            / f"{args.model_name}_l{args.layer}_llm_judge_2d_fixed_predictor.json"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, avg, predictor = load_components(args.checkpoint_path, device)
    predictor.eval()
    test_ids = sorted(int(x) for x in np.asarray(checkpoint["test_q_indices"]).reshape(-1))
    if args.max_samples is not None:
        test_ids = test_ids[: args.max_samples]

    dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    layer_name = f"model.layers.{args.layer}"
    cache = load_or_init_cache(args, args.model_dir)

    missing = []
    for q_idx in test_ids:
        question = dataset[q_idx]["question"]
        cached = cache["records"].get(str(q_idx))
        if not cache_record_is_valid(cached, q_idx, question):
            missing.append(q_idx)

    model = None
    tokenizer = None
    if missing:
        print(f"--- Pass 1/2 cache build: {len(missing)} missing of {len(test_ids)} ---")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        for q_idx in tqdm(missing, desc="Cache A0+h_last", unit="q"):
            question = dataset[q_idx]["question"]
            generation_prompt = format_generation_prompt(
                question,
                preset=args.preset,
                use_instruction=not args.no_instruction,
                model_name=args.model_name,
            )
            unsteered = generate_answer(
                model,
                tokenizer,
                question,
                hook=None,
                layer_name=layer_name,
                max_new_tokens=args.max_new_tokens,
                prompt_text=generation_prompt,
            )
            probe = capture_probe_last_hidden(
                model,
                tokenizer,
                question,
                unsteered,
                layer_name,
                prompt_text=generation_prompt,
            )
            cache["records"][str(q_idx)] = {
                "question_index": int(q_idx),
                "question": question,
                "unsteered_answer": unsteered,
                **probe,
            }
            _atomic_torch_save(cache, args.cache_path)
    else:
        print(f"Pass-1/2 cache complete: {len(test_ids)}/{len(test_ids)} questions")

    if args.cache_only:
        if model is not None:
            del model
            torch.cuda.empty_cache()
            gc.collect()
        print(f"Cache ready: {os.path.abspath(args.cache_path)}")
        return

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    prediction_hook = None
    dynamic_hook = None
    steering_stats = {"total": 0, "steered": 0, "prediction_count": 0}
    if not args.disable_steering:
        prediction_hook = get_2d_dynamic_intervention(
            avg,
            predictor,
            strength=args.strength,
            slerp_t=args.slerp_t,
        )
        dynamic_hook = get_2d_dynamic_intervention(
            avg,
            predictor,
            strength=args.strength,
            slerp_t=args.slerp_t,
            stats=steering_stats,
        )

    records = []
    for q_idx in tqdm(test_ids, desc="Dyn2D steered generation", unit="q"):
        cached = cache["records"][str(q_idx)]
        question = cached["question"]
        generation_prompt = format_generation_prompt(
            question,
            preset=args.preset,
            use_instruction=not args.no_instruction,
            model_name=args.model_name,
        )
        if args.disable_steering:
            steered = generate_answer(
                model,
                tokenizer,
                question,
                hook=None,
                layer_name=layer_name,
                max_new_tokens=args.max_new_tokens,
                prompt_text=generation_prompt,
            )
        else:
            fixed_controls = controls_from_cached_hidden(
                prediction_hook,
                predictor,
                cached["probe_last_hidden"],
                layer_name,
            )
            steered = generate_answer(
                model,
                tokenizer,
                question,
                hook=dynamic_hook,
                layer_name=layer_name,
                max_new_tokens=args.max_new_tokens,
                prompt_text=generation_prompt,
                predictor=predictor,
                fixed_controls=fixed_controls,
            )
        records.append(
            {
                "question_index": int(q_idx),
                "question": question,
                "probe": {
                    "last_token_id": int(cached["probe_last_token_id"]),
                    "last_token_text": cached["probe_last_token_text"],
                    "hidden_cached_in": os.path.abspath(args.cache_path),
                },
                "unsteered": {"answer": cached["unsteered_answer"]},
                "steered": {"answer": steered},
            }
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    truth_summary = run_judge_phase(args.truth_judge_path, records, info=False)
    info_summary = run_judge_phase(args.info_judge_path, records, info=True)
    summary = {
        "count": len(records),
        "truth": truth_summary,
        "info": info_summary,
        "truth_x_info": {
            "unsteered": (
                truth_summary["unsteered_mean_probability"]
                * info_summary["unsteered_mean_probability"]
            ),
            "steered": (
                truth_summary["steered_mean_probability"]
                * info_summary["steered_mean_probability"]
            ),
        },
    }
    for record in records:
        record["delta"] = {
            "truth_probability": (
                record["steered"].get("truth_probability", 0.0)
                - record["unsteered"].get("truth_probability", 0.0)
            ),
            "info_probability": (
                record["steered"].get("info_probability", 0.0)
                - record["unsteered"].get("info_probability", 0.0)
            ),
        }

    prediction_count = max(int(steering_stats.get("prediction_count", 0)), 1)
    output = {
        "method": "3-pass fixed-predictor Dyn2D",
        "model_name": args.model_name,
        "model_dir": args.model_dir,
        "layer": int(args.layer),
        "checkpoint_path": args.checkpoint_path,
        "cache_path": os.path.abspath(args.cache_path),
        "parameters": {
            "strength": float(args.strength),
            "slerp_t": float(args.slerp_t),
            "steering_enabled": not args.disable_steering,
            "max_new_tokens": int(args.max_new_tokens),
            "intervention": "h_new = h + strength * SLERP(V,R,t) * ||v_hat||_2",
        },
        "summary": summary,
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
        "results": records,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Truth: {truth_summary['unsteered_mean_probability']:.4f} -> "
        f"{truth_summary['steered_mean_probability']:.4f}"
    )
    print(
        f"Info: {info_summary['unsteered_mean_probability']:.4f} -> "
        f"{info_summary['steered_mean_probability']:.4f}"
    )
    print(f"Saved JSON: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
