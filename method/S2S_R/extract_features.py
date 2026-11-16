"""Extract raw answer-level hidden states and truthful/negative pairs for S2S_R.

The direct-MLP trainer consumes raw ``h_neg``/``h_pos`` pairs.  Training
normalizes only the MLP input and learns the full vector
``v_i = h_pos - h_neg`` directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.S2S_R.data import (
    HF_NAMES,
    build_question_vectors,
    get_truthfulqa_data,
)


def extract_last_token_activations(model, tokenizer, prompts, layer_idx: int, device: torch.device) -> np.ndarray:
    """Extract one last-token activation per prompt, matching the reference code."""
    activations = []
    model.eval()
    input_device = next(model.get_input_embeddings().parameters()).device
    from tqdm import tqdm
    with torch.inference_mode():
        for i, prompt in enumerate(tqdm(prompts, desc="Extracting features")):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = outputs.hidden_states[layer_idx + 1]
            last_token_act = hs[0, -1, :].detach().float().cpu().numpy()
            # 检查输出是否有问题
            # if np.isnan(last_token_act).any():
            #     print(f"NaN detected at prompt index: {i}")
            #     print("Prompt:", prompt[:300])
            #     print("NaN count:", int(np.isnan(last_token_act).sum()))
            # if not np.isfinite(last_token_act).all():
            #     print(f"Non-finite activation detected at prompt index: {i}")
            #     print("Inf count:", int(np.isinf(last_token_act).sum()))
            activations.append(last_token_act)
    activations = np.stack(activations, axis=0)
    if np.isnan(activations).any():
        print("Final activations contain NaN")
        print("Total NaN count:", int(np.isnan(activations).sum()))
        print("NaN positions:", np.argwhere(np.isnan(activations))[:20])
    else:
        print("No NaN found in activations")
    return activations


def main() -> None:
    p = argparse.ArgumentParser(description="Extract S2S_R paired hidden representations")
    p.add_argument("model_name")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--save_dir", default="./features")
    p.add_argument("--output_name", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Kept for compatibility; question K-fold ordering is deterministic and unsuffled.")
    args = p.parse_args()
    if args.layer < 0 or args.batch_size <= 0 or args.max_length < 0:
        raise ValueError("layer must be nonnegative, batch_size positive, max_length nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path} | device={device} | dtype={torch_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     trust_remote_code=False,
    #     device_map="auto",
    # )

    model.eval()
    data = get_truthfulqa_data(style="standard")
    activations = extract_last_token_activations(model, tokenizer, data["prompts"], args.layer, device)
    (
        vector_q_indices, positive_sample_indices, negative_sample_indices,
        positive_choice_indices, negative_choice_indices, vector_question_offsets,
        z_pos, z_neg, v_t, v_h, positive_counts, negative_counts, pair_counts,
    ) = build_question_vectors(activations, data["labels"], data["q_indices"], data["choice_indices"])
    # Keep both raw and normalized pair states for compatibility. The current
    # direct-MLP trainer uses the raw pair fields and normalizes h_neg itself.
    z_neg_raw, z_pos_raw = z_neg, z_pos
    z_neg = z_neg_raw / np.maximum(np.linalg.norm(z_neg_raw, axis=1, keepdims=True), 1e-12)
    z_pos = z_pos_raw / np.maximum(np.linalg.norm(z_pos_raw, axis=1, keepdims=True), 1e-12)
    os.makedirs(args.save_dir, exist_ok=True)
    safe = args.model_name.replace("/", "_").replace("\\", "_")
    name = args.output_name or f"{safe}_layer{args.layer}_S2S_R_pairs.npz"
    path = os.path.join(args.save_dir, name)
    np.savez_compressed(
        path,
        activations=activations, labels=data["labels"], q_indices=data["q_indices"],
        choice_indices=data["choice_indices"], prompts=data["prompts"], choices=data["choices"],
        questions=data["questions"], question_offsets=data["question_offsets"],
        split_seed=np.asarray(args.split_seed, dtype=np.int64),
        vector_q_indices=vector_q_indices, positive_sample_indices=positive_sample_indices,
        negative_sample_indices=negative_sample_indices, positive_choice_indices=positive_choice_indices,
        negative_choice_indices=negative_choice_indices, vector_question_offsets=vector_question_offsets,
        z_pos=z_pos, z_neg=z_neg, z_minus=z_neg, z_plus=z_pos,
        z_pos_raw=z_pos_raw, z_neg_raw=z_neg_raw, v_T=v_t, v_H=v_h,
        positive_counts=positive_counts, negative_counts=negative_counts, pair_counts=pair_counts,
        layer=np.asarray(args.layer, dtype=np.int64), model_name=np.asarray(args.model_name),
        model_path=np.asarray(model_path), dataset_name=np.asarray("truthful_qa"),
        dataset_config=np.asarray("multiple_choice"), target_name=np.asarray("mc2_targets"),
        method=np.asarray("S2S_R paired feature extraction"),
        vector_definition=np.asarray("v_i = z_pos_raw - z_neg_raw; direct MLP input is normalize(z_neg_raw)"),
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
