"""Evaluate both S2S grids/folds with one Llama load per layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from results import SELECTION, cells, file_identity, valid_cell, write_json


def evaluate_one(model, tokenizer, dataframe, config, layer, hook):
    import torch
    from tqdm import tqdm
    from method.S2S_R.eval_utils import calculate_scores_zeroshot, format_best, get_layer_name, split_multi_answer
    from truthfulqa.models import MC_calcs, set_columns

    results = dataframe.copy().reset_index(drop=True)
    tag = config["model_name"]
    set_columns(tag, results)
    for i in tqdm(range(len(results)), desc="TruthfulQA zero-shot", leave=False):
        row = results.iloc[i]
        true = split_multi_answer(row["Correct Answers"])
        false = split_multi_answer(row["Incorrect Answers"])
        common = dict(model=model, tokenizer=tokenizer, question=row["Question"],
                      base_hook_fn=hook, layer_name=get_layer_name(tag, layer),
                      device=torch.device(config["device"]),
                      use_instruction=True, model_name=tag)
        scores_true = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in true]
        scores_false = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in false]
        MC_calcs(tag, results, i, scores_true, scores_false, true, format_best(row["Best Answer"]))
    return {metric: float(results[f"{tag} {metric}"].mean()) for metric in ("MC1", "MC2", "MC3")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--checkpoint0", type=Path, required=True)
    parser.add_argument("--checkpoint1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    contexts = {}
    for fold, checkpoint in enumerate((args.checkpoint0, args.checkpoint1)):
        contexts[fold] = {"layer": args.layer, "fold": fold, "checkpoint": file_identity(checkpoint),
                          **{k: config[k] for k in ("model_name", "model_dir", "device", "dtype", "seed", "code")},
                          "eval_split": "test", "prompt": "zero_shot_instruction"}
    pending = []
    for method, parameters, fold, relative in cells(config, args.layer):
        path = args.output / relative
        report = None
        if args.resume:
            try:
                report = json.loads(path.read_text())
            except (OSError, ValueError):
                pass
        if isinstance(report, dict) and valid_cell(report, contexts[fold], method, parameters):
            print(f"[reuse] {method} layer={args.layer} fold={fold} {parameters}", flush=True)
        else:
            pending.append((method, parameters, fold, path))
    if not pending:
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from method.S2S_L.steering import get_s2s_l_intervention
    from method.S2S_R.evaluate_S2S_R import get_s2s_r_intervention
    from method.S2S_R.evaluate_predictor import load_components
    from method.S2S_R.eval_utils import load_truthfulqa_mc_dataframe, resolve_device, set_seed

    device = resolve_device(config["device"])
    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(str(ROOT / "TruthfulQA/data/v1/TruthfulQA.csv"))
    components, frames, test_ids = {}, {}, {}
    for fold, checkpoint in enumerate((args.checkpoint0, args.checkpoint1)):
        data, avg, predictor = load_components(str(checkpoint), device)
        try:
            if int(data["fold_idx"]) != fold or int(data["num_folds"]) != 2:
                raise ValueError("Checkpoint fold mismatch")
            train = set(map(int, data["train_q_indices"]))
            test = set(map(int, data["test_q_indices"]))
            if not train or not test or train & test or train | test != set(range(len(hf_dataset))):
                raise ValueError("Invalid outer train/test partition")
            test_ids[fold] = sorted(test)
        finally:
            data.close()
        frames[fold] = dataframe[dataframe["hf_idx"].isin(test)].copy().reset_index(drop=True)
        if len(frames[fold]) != len(test):
            raise ValueError("CSV/HF held-out question mapping is incomplete")
        components[fold] = (avg, predictor)
    if set(test_ids[0]) & set(test_ids[1]) or set(test_ids[0]) | set(test_ids[1]) != set(range(len(hf_dataset))):
        raise ValueError("Fold test partitions must be disjoint and cover TruthfulQA")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    dtype = torch.float32 if device.type == "cpu" else getattr(torch, config["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"], torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=None if device.type == "cpu" else "auto", trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for method, parameters, fold, path in pending:
        set_seed(config["seed"])
        stats = {"total": 0, "steered": 0, "weight_count": 0, "prediction_count": 0}
        factory = get_s2s_l_intervention if method == "S2S-L" else get_s2s_r_intervention
        hook = factory(*components[fold], **parameters, stats=stats)
        print(f"[evaluate] {method} layer={args.layer} fold={fold} {parameters}", flush=True)
        metrics = evaluate_one(model, tokenizer, frames[fold], config, args.layer, hook)
        report = {"selection": SELECTION, "context": contexts[fold], "method": method,
                  "parameters": parameters, "test_q_indices": test_ids[fold],
                  "n_questions": len(frames[fold]), "metrics": metrics,
                  "stats": {k: int(stats.get(k, 0)) for k in ("total", "steered", "prediction_count")}}
        if not valid_cell(report, contexts[fold], method, parameters):
            raise ValueError(f"Invalid metrics: {metrics}")
        write_json(path, report)
        print(f"[saved] {metrics}", flush=True)


if __name__ == "__main__":
    main()
