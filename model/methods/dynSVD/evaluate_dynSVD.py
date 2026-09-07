from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.methods.dynSVD.dynSVD import (
    ResidualRouter,
    ScaleNet,
    normalize_basis_combination,
)
from model.methods.static.evaluate_onlysteer import (
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
    data = np.load(path, allow_pickle=False)
    dim = int(data["hidden_dim"].item())
    rank = int(data["rank"].item())
    router_hidden = int(data["router_hidden"].item())

    avg = torch.as_tensor(data["v_avg"], dtype=torch.float32, device=device)
    basis = torch.as_tensor(data["basis"], dtype=torch.float32, device=device)
    if avg.shape != (dim,) or basis.shape != (rank, dim):
        raise ValueError("checkpoint v_avg/basis shape mismatch")

    shared_fc1 = torch.nn.Linear(dim, router_hidden).to(device)
    router = ResidualRouter(dim, rank, router_hidden, shared_fc1).to(device)
    state = {
        "fc1.weight": torch.as_tensor(data["router_fc1.weight"]),
        "fc1.bias": torch.as_tensor(data["router_fc1.bias"]),
        "fc2.weight": torch.as_tensor(data["router_fc2.weight"]),
        "fc2.bias": torch.as_tensor(data["router_fc2.bias"]),
    }
    router.load_state_dict(state)
    if not all(k in data.files for k in ("scale_fc1.weight", "scale_fc1.bias", "scale_fc2.weight", "scale_fc2.bias")):
        raise KeyError("checkpoint missing ScaleNet weights; retrain dynSVD checkpoint")
    scale_net = ScaleNet(dim, router_hidden, shared_fc1).to(device)
    scale_state = {
        "fc1.weight": torch.as_tensor(data["scale_fc1.weight"]),
        "fc1.bias": torch.as_tensor(data["scale_fc1.bias"]),
        "fc2.weight": torch.as_tensor(data["scale_fc2.weight"]),
        "fc2.bias": torch.as_tensor(data["scale_fc2.bias"]),
    }
    scale_net.load_state_dict(scale_state)
    scale_net.eval()
    router.eval()
    return data, avg, basis, router, scale_net


def get_dynamic_intervention(
    avg,
    basis,
    router,
    scale_net,
    alpha: float,
    gamma: float = 1.0,
    hidden_cosine_threshold: float = 0.99,
    stats: dict | None = None,
):
    """Create a hidden-state-cosine-gated dynSVD intervention hook."""
    rank = int(basis.shape[0])
    # Cosine values themselves lie in [-1, 1], but thresholds outside that
    # interval are useful controls: a threshold > 1 always steers, while a
    # threshold < -1 always skips. Accept any finite value.
    if not np.isfinite(hidden_cosine_threshold):
        raise ValueError("hidden_cosine_threshold must be finite")

    def hook(output, layer_name=None, start_idx: int | None = None):
        hidden, rest = (
            (output[0], output[1:])
            if isinstance(output, tuple)
            else (output, None)
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

        # Use the pre-intervention hidden state at its original scale. This
        # must match the raw z- representations used to train the router.
        z = hidden[:, start, :].float()
        logits = router.logits(z)
        scale = scale_net(z)
        coeff = torch.softmax(logits, dim=-1)
        # The eval-time residual is direction-only. The training-time
        # ScaleNet is intentionally not used here; gamma controls the fixed
        # residual magnitude for this evaluation run.
        routed_direction = normalize_basis_combination(coeff, basis)
        residual = routed_direction
        dynamic = avg.unsqueeze(0).expand(z.shape[0], -1)
        original = hidden[:, start:, :].float()
        steering = (
            alpha * dynamic[:, None, :]
            + gamma * residual[:, None, :]
        )
        # steering = alpha * (dynamic[:, None, :]+ scale * residual[:, None, :])
        print(scale)
        candidate = original + steering
        # Mean token-wise cosine between the unsteered and candidate hidden
        # states. High similarity means the intervention is negligible and is
        # therefore skipped; only lower-similarity candidates are applied.
        hidden_cosine = F.cosine_similarity(candidate, original, dim=-1).mean(dim=-1)
        gate = hidden_cosine < hidden_cosine_threshold
        gate_scale = gate.to(dynamic.dtype)[:, None, None]

        edited = hidden.clone()
        edited[:, start:, :] = (
            hidden[:, start:, :].float()
            + steering * gate_scale
        ).to(hidden.dtype)

        if stats is not None:
            n_tokens = hidden.shape[1] - start
            hidden_cosine_cpu = hidden_cosine.detach().cpu()
            residual_norm_cpu = torch.linalg.vector_norm(residual, dim=-1).detach().cpu()
            gate_cpu = gate.detach().cpu()

            stats["total"] = stats.get("total", 0) + hidden.shape[0] * n_tokens
            stats["steered"] = (
                stats.get("steered", 0) + int(gate.sum().item()) * n_tokens
            )
            stats["hidden_cosine_sum"] = stats.get("hidden_cosine_sum", 0.0) + float(hidden_cosine_cpu.sum())
            stats["hidden_cosine_count"] = stats.get("hidden_cosine_count", 0) + int(hidden_cosine.numel())
            stats["batch_hidden_cosine_mean"] = float(hidden_cosine_cpu.mean())
            stats["batch_steering_rate"] = float(gate_cpu.float().mean())

            residual_log = stats.get("residual_log")
            if residual_log is not None:
                batch_index = int(stats.get("residual_batch_count", 0))
                record = {
                    "batch_index": batch_index,
                    "question_index": stats.get("question_index"),
                    "question": stats.get("question"),
                    "start_idx": start,
                    "batch_size": int(hidden_cosine.numel()),
                    "residual_norm": residual_norm_cpu.tolist(),
                    "hidden_cosine": hidden_cosine_cpu.tolist(),
                    "hidden_cosine_mean": float(hidden_cosine_cpu.mean()),
                    "steer": gate_cpu.tolist(),
                    "threshold": float(hidden_cosine_threshold),
                }
                residual_log.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                residual_log.flush()
                stats["residual_batch_count"] = batch_index + 1

        return (edited,) + rest if rest is not None else edited

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dynSVD")
    parser.add_argument("model_name")
    parser.add_argument(
        "--checkpoint_path",
        "--operator_path",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--csv_path",
        default="./TruthfulQA/data/v1/TruthfulQA.csv",
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument(
        "--eval_split",
        choices=["test", "train", "all"],
        default="test",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0, help="Fixed magnitude applied to the routed residual direction during eval (ScaleNet is not used).")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda"
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--hidden_cosine_threshold", type=float, default=0.99, help="Skip steering when cosine(original hidden, candidate hidden) is at or above this threshold.")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--residual_log_path", default=None, help="JSONL path for per-batch residual norms.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    data, avg, basis, router, scale_net = load_components(args.checkpoint_path, device)
    alpha = 1.0 if args.alpha is None else args.alpha
    if alpha == 0:
        raise ValueError("alpha must be nonzero")
    if not np.isfinite(args.gamma):
        raise ValueError("gamma must be finite")
    print(f"dynSVD | dim={basis.shape[1]} rank={basis.shape[0]} alpha={alpha} gamma={args.gamma} hidden_cosine_threshold={args.hidden_cosine_threshold}")
    model_path = args.model_dir or HF_NAMES.get(
        args.model_name, args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    if device.type == "cpu":
        dtype = torch.float32
        device_map = None
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    df, hf_dataset = load_truthfulqa_mc_dataframe(args.csv_path)
    if args.eval_split == "all":
        ids = set(range(len(hf_dataset)))
    else:
        key = (
            "train_q_indices"
            if args.eval_split == "train"
            else "test_q_indices"
        )
        if key not in data.files:
            raise KeyError(f"checkpoint missing {key}")
        ids = set(int(x) for x in np.asarray(data[key]).reshape(-1))

    results = df[df["hf_idx"].isin(ids)].copy().reset_index(drop=True)
    tag = args.model_name
    set_columns(tag, results)
    mode = "few_shot" if args.few_shot else "zero_shot"
    layer_name = get_layer_name(args.model_name, args.layer)

    if args.output_path is None:
        args.output_path = (
            f"results/{args.model_name.replace('/', '_')}"
            f"_l{args.layer}_dynSVD_{mode}.json"
        )
    if args.residual_log_path is None:
        args.residual_log_path = args.output_path.replace(
            ".json", "_residual_norm.jsonl"
        )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.residual_log_path).parent.mkdir(
        parents=True, exist_ok=True
    )
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
        else get_dynamic_intervention(
            avg,
            basis,
            router,
            scale_net,
            alpha,
            args.gamma,
            args.hidden_cosine_threshold,
            stats,
        )
    )

    try:
        pbar = tqdm(range(len(results)), desc=mode)
        for i in pbar:
            row = results.iloc[i]
            stats["question_index"] = int(i)
            stats["question"] = row["Question"]

            true = split_multi_answer(row["Correct Answers"])
            false = split_multi_answer(row["Incorrect Answers"])
            kwargs = dict(
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
                st = calculate_scores_fewshot(
                    choices=true, preset=args.preset, **kwargs
                )
                sf = calculate_scores_fewshot(
                    choices=false, preset=args.preset, **kwargs
                )
            else:
                st = [
                    calculate_scores_zeroshot(
                        choices=[answer], **kwargs
                    )[0]
                    for answer in true
                ]
                sf = [
                    calculate_scores_zeroshot(
                        choices=[answer], **kwargs
                    )[0]
                    for answer in false
                ]

            MC_calcs(
                tag,
                results,
                i,
                st,
                sf,
                true,
                format_best(row["Best Answer"]),
            )
            running = {
                metric: float(
                    results[f"{tag} {metric}"].iloc[: i + 1].mean()
                )
                for metric in ("MC1", "MC2", "MC3")
            }
            pbar.set_description(
                f"MC1:{running['MC1']:.3f} "
                f"MC2:{running['MC2']:.3f} "
                f"MC3:{running['MC3']:.3f} "
                f"C_batch:{stats.get('batch_hidden_cosine_mean', float('nan')):.3f} "
                f"steer:{stats.get('batch_steering_rate', float('nan')):.3f}"
            )
    finally:
        residual_log.close()

    metrics = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    metrics.update(
        {
            "gamma": args.gamma,
            "hidden_cosine_threshold": args.hidden_cosine_threshold,
            "mean_hidden_cosine": stats.get("hidden_cosine_sum", 0.0)
            / max(stats.get("hidden_cosine_count", 0), 1),
            "steering_rate": stats.get("steered", 0)
            / max(stats.get("total", 0), 1),
            "residual_log_path": args.residual_log_path,
        }
    )
    print(
        f"MC1: {metrics['MC1']:.4f} | "
        f"MC2: {metrics['MC2']:.4f} | "
        f"MC3: {metrics['MC3']:.4f} | "
        f"hidden_cosine: {metrics['mean_hidden_cosine']:.4f} | "
        f"steer_rate: {metrics['steering_rate']:.4f}"
    )

    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "layer": args.layer,
                "checkpoint_path": args.checkpoint_path,
                "alpha": alpha,
                "gamma": args.gamma,
                "eval_split": args.eval_split,
                "metrics": metrics,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {args.output_path}")
    if args.save_details:
        results.to_csv(
            args.output_path.replace(".json", "_details.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
