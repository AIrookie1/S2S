"""Load/evaluate S2S_R's direct MLP vector predictor."""
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

from methods.S2S_R.predictor import DirectVectorMLP, predict_residual
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
from truthfulqa.models import MC_calcs, set_columns


def load_components(path: str, device: torch.device):
    """Load ``v_avg`` and the direct vector MLP from a S2S_R checkpoint."""
    data = np.load(path, allow_pickle=False)
    required = {
        "hidden_dim",
        "mlp_hidden",
        "v_avg",
        "mlp_fc1.weight",
        "mlp_fc1.bias",
        "mlp_fc2.weight",
        "mlp_fc2.bias",
    }
    missing = required.difference(data.files)
    if missing:
        data.close()
        raise KeyError(
            "S2S_R direct-MLP checkpoint missing fields: " + ", ".join(sorted(missing))
        )

    dim = int(np.asarray(data["hidden_dim"]).item())
    hidden = int(np.asarray(data["mlp_hidden"]).item())
    avg = torch.as_tensor(data["v_avg"], dtype=torch.float32, device=device)
    if avg.shape != (dim,):
        data.close()
        raise ValueError(
            f"checkpoint v_avg shape mismatch: expected {(dim,)}, got {tuple(avg.shape)}"
        )

    predictor = DirectVectorMLP(dim, hidden, dim).to(device)
    predictor.load_state_dict(
        {
            "fc1.weight": torch.as_tensor(data["mlp_fc1.weight"], dtype=torch.float32),
            "fc1.bias": torch.as_tensor(data["mlp_fc1.bias"], dtype=torch.float32),
            "fc2.weight": torch.as_tensor(data["mlp_fc2.weight"], dtype=torch.float32),
            "fc2.bias": torch.as_tensor(data["mlp_fc2.bias"], dtype=torch.float32),
        }
    )
    predictor.eval()
    return data, avg, predictor


def get_dynamic_intervention(
    avg: torch.Tensor,
    predictor: DirectVectorMLP,
    alpha: float,
    gamma: float = 1.0,
    stats: dict | None = None,
):
    """Add ``alpha*v_avg + gamma*normalize(v_hat-v_avg)`` to the segment."""

    def hook(output, layer_name=None, start_idx: int | None = None):
        del layer_name
        hidden, rest = (
            (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        )
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        start = max(
            0,
            min(
                int(start_idx if start_idx is not None else hidden.shape[1] - 1),
                hidden.shape[1] - 1,
            ),
        )
        probe = hidden[:, start, :].float()
        v_hat, residual = predict_residual(predictor, probe, avg)
        original = hidden[:, start:, :].float()
        steering = alpha * avg.view(1, 1, -1) + gamma * residual[:, None, :]
        candidate = original + steering
        edited = hidden.clone()
        edited[:, start:, :] = candidate.to(hidden.dtype)

        if stats is not None:
            n_tokens = hidden.shape[0] * (hidden.shape[1] - start)
            stats["total"] = stats.get("total", 0) + n_tokens
            stats["steered"] = stats.get("steered", 0) + n_tokens
            residual_log = stats.get("residual_log")
            if residual_log is not None:
                record = {
                    "batch_index": int(stats.get("residual_batch_count", 0)),
                    "question_index": stats.get("question_index"),
                    "question": stats.get("question"),
                    "start_idx": start,
                    "batch_size": int(hidden.shape[0]),
                    "predicted_v_norm": torch.linalg.vector_norm(v_hat, dim=-1)
                    .detach()
                    .cpu()
                    .tolist(),
                    "residual_norm": torch.linalg.vector_norm(residual, dim=-1)
                    .detach()
                    .cpu()
                    .tolist(),
                }
                residual_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                residual_log.flush()
                stats["residual_batch_count"] = record["batch_index"] + 1

        return (edited,) + rest if rest is not None else edited

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S2S_R direct-MLP steering")
    parser.add_argument("model_name")
    parser.add_argument(
        "--checkpoint_path",
        "--operator_path",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./data/truthfulqa/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--residual_log_path", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    data, avg, predictor = load_components(args.checkpoint_path, device)
    alpha = 1.0 if args.alpha is None else args.alpha
    if alpha == 0:
        raise ValueError("alpha must be nonzero")
    if not np.isfinite(args.gamma):
        raise ValueError("gamma must be finite")
    print(
        f"S2S_R direct-MLP | dim={avg.numel()} hidden={predictor.hidden_dim} "
        f"alpha={alpha} gamma={args.gamma}"
    )

    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device.type == "cpu":
        dtype, device_map = torch.float32, None
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
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
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataframe, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if args.eval_split == "all":
        ids = set(range(len(hf_dataset)))
    else:
        key = "train_q_indices" if args.eval_split == "train" else "test_q_indices"
        if key not in data.files:
            raise KeyError(f"checkpoint missing {key}")
        ids = set(int(x) for x in np.asarray(data[key]).reshape(-1))

    results = dataframe[dataframe["hf_idx"].isin(ids)].copy().reset_index(drop=True)
    tag = args.model_name
    set_columns(tag, results)
    mode = "few_shot" if args.few_shot else "zero_shot"
    layer_name = get_layer_name(args.model_name, args.layer)
    if args.output_path is None:
        args.output_path = (
            f"results/{args.model_name.replace('/', '_')}_l{args.layer}_predictor_{mode}.json"
        )
    if args.residual_log_path is None:
        args.residual_log_path = args.output_path.replace(".json", "_residual_norm.jsonl")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.residual_log_path).parent.mkdir(parents=True, exist_ok=True)

    residual_log = open(args.residual_log_path, "w", encoding="utf-8", buffering=1)
    stats = {
        "total": 0,
        "steered": 0,
        "residual_log": residual_log,
        "residual_batch_count": 0,
    }
    hook = (
        None
        if args.no_intervention
        else get_dynamic_intervention(avg, predictor, alpha, args.gamma, stats)
    )

    try:
        progress = tqdm(range(len(results)), desc=mode)
        for i in progress:
            row = results.iloc[i]
            stats["question_index"] = int(i)
            stats["question"] = row["Question"]
            true_answers = split_multi_answer(row["Correct Answers"])
            false_answers = split_multi_answer(row["Incorrect Answers"])
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
                scores_true = calculate_scores_fewshot(
                    choices=true_answers, preset=args.preset, **common
                )
                scores_false = calculate_scores_fewshot(
                    choices=false_answers, preset=args.preset, **common
                )
            else:
                scores_true = [
                    calculate_scores_zeroshot(choices=[answer], **common)[0]
                    for answer in true_answers
                ]
                scores_false = [
                    calculate_scores_zeroshot(choices=[answer], **common)[0]
                    for answer in false_answers
                ]
            MC_calcs(
                tag,
                results,
                i,
                scores_true,
                scores_false,
                true_answers,
                format_best(row["Best Answer"]),
            )
            running = {
                metric: float(results[f"{tag} {metric}"].iloc[: i + 1].mean())
                for metric in ("MC1", "MC2", "MC3")
            }
            progress.set_description(
                f"MC1:{running['MC1']:.3f} MC2:{running['MC2']:.3f} "
                f"MC3:{running['MC3']:.3f}"
            )
            if args.save_details:
                results.loc[i, f"{tag} scores_true"] = str(scores_true)
                results.loc[i, f"{tag} scores_false"] = str(scores_false)
    finally:
        residual_log.close()

    metrics = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    report = {
        "method": "S2S_R direct MLP additive steering",
        "model_name": args.model_name,
        "layer": int(args.layer),
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "parameters": {"alpha": alpha, "gamma": args.gamma},
        "metrics": metrics,
        "results": json.loads(results.to_json(orient="records", force_ascii=False)),
    }
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
