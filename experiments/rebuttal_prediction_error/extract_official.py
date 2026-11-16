"""Extract labeled official evaluation splits with the released task prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal_prediction_error.rebuttal_prediction_error_run import LAYERS, MODEL
from method.S2S_R.generic.prompts_generic import (
    append_option, format_boolq_prompt, format_copa_prompt,
)
from method.S2S_R.generic.utils_generic import get_dataset_data, set_seed

SPLITS = {"boolq": "validation", "copa": "validation", "storycloze": "eval",
          "winogrande": "validation", "mmlu_global": "test_category_rows_500_1000"}


def official_data(dataset, tokenizer):
    # The legacy unified feature loader ignores split='test' for BoolQ/COPA.
    # Load those validation sets explicitly to prevent training-set leakage.
    if dataset in ("boolq", "copa"):
        data = load_dataset("super_glue", dataset, split="validation")
        prompts, labels, questions = [], [], []
        for i, row in enumerate(data):
            if dataset == "boolq":
                base = format_boolq_prompt(row["passage"], row["question"])
                options = ["no", "yes"]
            else:
                options = [row["choice1"], row["choice2"]]
                base = format_copa_prompt(row["premise"], row["question"], *options)
            correct = int(row["label"])
            if correct not in (0, 1):
                raise ValueError("Evaluation split has missing labels")
            for j, option in enumerate(options):
                prompts.append(append_option(base, option))
                labels.append(int(j == correct))
                questions.append(i)
        return prompts, np.array(labels), np.array(questions)
    return get_dataset_data(dataset, split="test", seed=42, tokenizer=tokenizer)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model-dir", default=__import__("os").environ.get("LLAMA_MODEL_DIR", "models/Meta-Llama-3.1-8B-Instruct"))
    p.add_argument("--output", type=Path, default=ROOT / "results/experiment/rebuttal_prediction_error/official/features")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--datasets", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    p.add_argument("--audit-only", action="store_true", help="Check cached training activations against current extraction")
    args = p.parse_args()
    torch.set_num_threads(4)
    set_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [d for d in args.datasets if not (args.output / f"{d}_official_l{LAYERS[d]}.npz").exists()]
    if not pending and not args.audit_only:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16,
                                               device_map={"": "cuda:0"})
    model.eval()
    if args.audit_only or pending:
        audits = []
        for dataset in args.datasets:
            layer = LAYERS[dataset]
            prompts, _, _ = get_dataset_data(dataset, split="train", seed=42, tokenizer=tokenizer)
            ids = np.linspace(0, len(prompts) - 1, 6, dtype=int)
            with np.load(ROOT / f"results/reproduction/generic/features/S2S_R/{MODEL}/{MODEL}_{dataset}_train_l{layer}.npz") as cached:
                reference = torch.tensor(cached["activations"][ids], dtype=torch.float32)
            inputs = tokenizer([prompts[i] for i in ids], padding=True,
                add_special_tokens=False, return_tensors="pt").to("cuda:0")
            with torch.inference_mode(), TraceDict(model.model, [f"layers.{layer}"], stop=True) as trace:
                model.model(**inputs, use_cache=False)
            hidden = trace[f"layers.{layer}"].output
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            current = hidden[torch.arange(len(ids), device=hidden.device), inputs["attention_mask"].sum(1) - 1].float().cpu()
            cosine = torch.nn.functional.cosine_similarity(current, reference, dim=1)
            relative_l2 = (current - reference).norm(dim=1) / reference.norm(dim=1)
            audits.append({"dataset": dataset, "layer": layer, "sample_rows": ids.tolist(),
                "cosine_min": cosine.min().item(), "relative_l2_max": relative_l2.max().item(),
                "passed": bool((cosine > 0.995).all() and (relative_l2 < 0.02).all())})
            print(audits[-1], flush=True)
        (args.output.parent / "training_feature_audit.json").write_text(json.dumps(audits, indent=2))
        if not all(row["passed"] for row in audits):
            raise ValueError("Cached training feature protocol differs; inspect training_feature_audit.json")
        if args.audit_only:
            return
    for dataset in pending:
        prompts, labels, q = official_data(dataset, tokenizer)
        if not prompts or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"Invalid official data for {dataset}")
        layer = LAYERS[dataset]
        activations = np.empty((len(prompts), model.config.hidden_size), dtype=np.float16)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        print(f"{dataset}: official {SPLITS[dataset]}, {len(np.unique(q))} questions, {len(prompts)} candidates", flush=True)
        with torch.inference_mode():
            for start in tqdm(range(0, len(order), args.batch_size), desc=dataset):
                idx = order[start:start + args.batch_size]
                inputs = tokenizer([prompts[i] for i in idx], padding=True,
                                   add_special_tokens=False, return_tensors="pt").to("cuda:0")
                layer_name = f"layers.{layer}"
                # Stop after the required block; later blocks and LM logits cannot
                # change the captured hidden state and need not be computed.
                with TraceDict(model.model, [layer_name], stop=True) as trace:
                    model.model(**inputs, use_cache=False)
                hidden = trace[layer_name].output
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                last = inputs["attention_mask"].sum(1) - 1
                activations[idx] = hidden[torch.arange(len(idx), device=hidden.device), last].cpu().numpy()
        path = args.output / f"{dataset}_official_l{layer}.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, activations=activations, labels=labels, q_indices=q,
            dataset=np.asarray(dataset), source_split=np.asarray(SPLITS[dataset]),
            layer=np.asarray(layer), model_name=np.asarray(MODEL),
            model_path=np.asarray(args.model_dir), dtype=np.asarray("float16"),
            seed=np.asarray(42), prompt_definition=np.asarray("released S2S_R.generic prompts; option appended; add_special_tokens=False"))
        temp.replace(path)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
