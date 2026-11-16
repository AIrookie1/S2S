"""Three-vector spherical S2S_R steering with a direct MLP vector predictor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.S2S_R.predictor import predict_residual
from method.S2S_R.evaluate_predictor import load_components
from method.S2S_R.eval_utils import (
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
from method.inference_residual_noise import make_noise_generator, perturb_unit_direction
from truthfulqa.models import MC_calcs, set_columns


def norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def adaptive_3way_weights(
    x: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    kappa: float = 5.0,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return Similarity-Graph Adaptive 2-Slerp weights.

    ``x`` and ``v`` are unit vectors. ``r`` is the L2-normalized direct-MLP
    residual ``v_hat - v_avg``.
    """
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("weight kappa must be finite and positive")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("weight temperature must be finite and positive")

    s_xv = (x * v).sum(dim=-1, keepdim=True)
    s_xr = (x * r).sum(dim=-1, keepdim=True)
    s_vr = (v * r).sum(dim=-1, keepdim=True)

    temperature = kappa
    attn_x = torch.softmax(kappa * torch.cat([s_xv, s_xr], dim=-1), dim=-1)
    a_xv, a_xr = attn_x[..., 0:1], attn_x[..., 1:2]
    attn_v = torch.softmax(kappa * torch.cat([s_xv, s_vr], dim=-1), dim=-1)
    a_vx, a_vr = attn_v[..., 0:1], attn_v[..., 1:2]
    attn_r = torch.softmax(kappa * torch.cat([s_xr, s_vr], dim=-1), dim=-1)
    a_rx, a_rv = attn_r[..., 0:1], attn_r[..., 1:2]

    # Inject the directional prior before the final centrality softmax:
    # current hidden state X=0.5, mean direction V=0.25, residual R=0.25.
    # The resulting alpha/beta/gamma are then used consistently by every
    # ordering in the symmetric 2-Slerp composition below.
    centrality = torch.cat(
        [
            (a_vx + a_rx),
            (a_xv + a_rv),
            (a_xr + a_vr),
        ],
        dim=-1,
    )
    weights = torch.softmax((centrality) * temperature, dim=-1)
    return (
        weights[..., 0:1],
        weights[..., 1:2],
        weights[..., 2:3],
        {
            "s_xv": s_xv,
            "s_xr": s_xr,
            "s_vr": s_vr,
            "centrality": centrality,
            "weights": weights,
        },
    )


def slerp(x, y, t, eps=1e-8, small_angle_threshold=1e-4):
    dot = (x * y).sum(-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t_tensor = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    while t_tensor.ndim < theta.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
    result = (
        torch.sin((1 - t_tensor) * theta) / sin_theta.clamp_min(eps) * x
        + torch.sin(t_tensor * theta) / sin_theta.clamp_min(eps) * y
    )
    return result


def two_slerp(a, b, c, wa, wb, wc, eps, small_angle_threshold):
    denominator = torch.as_tensor(wa + wb, dtype=a.dtype, device=a.device)
    safe_denominator = denominator.clamp_min(eps)
    first = slerp(
        slerp(a, b, torch.as_tensor(wb, dtype=a.dtype, device=a.device) / safe_denominator, eps, small_angle_threshold),
        c,
        wc,
        eps,
        small_angle_threshold,
    )
    return torch.where(denominator > eps, first, c)


def two_slerp_sym(a, b, c, alpha, beta, gamma, eps=1e-8, small_angle_threshold=1e-4):
    first = two_slerp(a, b, c, alpha, beta, gamma, eps, small_angle_threshold)
    second = two_slerp(b, c, a, beta, gamma, alpha, eps, small_angle_threshold)
    third = two_slerp(c, a, b, gamma, alpha, beta, eps, small_angle_threshold)

    # first = torch.ones_like(first)
    # third = torch.ones_like(third)

    return norm(alpha * first + beta * second + gamma * third, eps)#


def validate_weights(alpha, beta, gamma):
    if not all(np.isfinite(x) and x >= 0 for x in (alpha, beta, gamma)):
        raise ValueError("alpha, beta, gamma must be finite and non-negative")
    if not np.isclose(alpha + beta + gamma, 1.0, atol=1e-6):
        raise ValueError(
            f"alpha + beta + gamma must equal 1 (got {alpha + beta + gamma:.8f})"
        )


def get_s2s_r_intervention(
    avg,
    predictor,
    weight_kappa=8.0,
    margin=-0.3,
    eps=1e-8,
    small_angle_threshold=1e-4,
    stats=None,
    weight_histogram_gate="all",
    method="S2S_R",
    static_weights=(0.6, 0.3, 0.1),
    residual_noise_sigma=0.0,
    residual_noise_seed=42,
):
    """Build the S2S_R rotation hook with adaptive-weight margin gating.

    A token is steered only when one of the directional components (the
    learned mean ``beta`` or predicted residual ``gamma``) exceeds the current
    hidden-state component ``alpha`` by ``margin``.  This keeps tokens whose
    alpha weight is already dominant unchanged.
    """
    # if not np.isfinite(margin) or margin < 0:
    #     raise ValueError("margin must be finite and non-negative")
    if method not in {"S2S_R", "S2S_R_Static"}:
        raise ValueError(f"Unsupported S2S_R method: {method}")
    if method == "S2S_R_Static":
        validate_weights(*static_weights)
    if not np.isfinite(residual_noise_sigma) or residual_noise_sigma < 0:
        raise ValueError("residual_noise_sigma must be finite and non-negative")
    if weight_histogram_gate not in {"all", "steered"}:
        raise ValueError(
            "weight_histogram_gate must be 'all' or 'steered' "
            f"(got {weight_histogram_gate!r})"
        )
    mean_direction = norm(avg, eps)
    noise_generator = None

    def hook(output, layer_name=None, start_idx=None):
        nonlocal noise_generator
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
        original = hidden[:, start:, :].float()
        original_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True).clamp_min(eps)
        # Preserve the original sequence-level S2S_R behavior: normalize the
        # current segment, predict one v_i from its last position, construct
        # residual = normalize(v_hat_i - v_avg), then broadcast that same
        # residual across the whole intervention segment.
        a = norm(original, eps)
        v_hat, residual = predict_residual(predictor, a[:, -1, :], avg, eps=eps)
        if residual_noise_sigma > 0:
            if noise_generator is None:
                noise_generator = make_noise_generator(residual_noise_seed, v_hat.device)
            residual = perturb_unit_direction(
                v_hat - avg,
                sigma=residual_noise_sigma,
                generator=noise_generator,
                eps=eps,
            ).float()
        residual = residual.float()
        b = mean_direction.view(1, 1, -1).expand_as(a)
        c = residual.view(residual.shape[0], 1, -1).expand_as(a)
        if method == "S2S_R":
            alpha, beta, gamma, weight_info = adaptive_3way_weights(
                a, b, c, kappa=weight_kappa, temperature=1.0
            )
        else:
            alpha = torch.full_like(a[..., :1], float(static_weights[0]))
            beta = torch.full_like(a[..., :1], float(static_weights[1]))
            gamma = torch.full_like(a[..., :1], float(static_weights[2]))
            weight_info = {"weights": torch.cat([alpha, beta, gamma], dim=-1)}

        rotated_direction = two_slerp_sym(
            a, b, c, alpha, beta, gamma, eps, small_angle_threshold
        )
        candidate = rotated_direction * original_norm
        should_steer = (
            torch.maximum(beta, gamma) > (alpha + margin)
            if method == "S2S_R" else torch.ones_like(alpha, dtype=torch.bool)
        )
        edited_segment = torch.where(should_steer, candidate, original)
        edited = hidden.clone()
        edited[:, start:, :] = edited_segment.to(hidden.dtype)
        if stats is not None:
            token_count = int(original.shape[0] * original.shape[1])
            steered_count = int(should_steer.sum().item())
            stats["total"] = stats.get("total", 0) + token_count
            stats["steered"] = stats.get("steered", 0) + steered_count
            stats["last_batch_rate"] = steered_count / max(token_count, 1)
            stats["steer_token_ratio"] = stats["steered"] / max(stats["total"], 1)
            stats["weight_count"] = stats.get("weight_count", 0) + token_count
            stats["alpha_sum"] = stats.get("alpha_sum", 0.0) + float(alpha.sum().item())
            stats["beta_sum"] = stats.get("beta_sum", 0.0) + float(beta.sum().item())
            stats["gamma_sum"] = stats.get("gamma_sum", 0.0) + float(gamma.sum().item())
            stats["last_weight_mean"] = weight_info["weights"].mean(dim=tuple(range(weight_info["weights"].ndim - 1))).detach().cpu().tolist()
            # Optional token-level ternary histogram. Rows index beta bins and
            # columns index gamma bins; alpha is implicit as 1-beta-gamma.
            # The histogram can use every token or only tokens that pass the
            # margin gate.  Main evaluation metrics above always keep their
            # original all-token adaptive-weight mean for backward compatibility.
            if "weight_histogram" in stats:
                bins = int(stats["weight_histogram_bins"])
                flat = weight_info["weights"].detach().float().reshape(-1, 3)
                if weight_histogram_gate == "steered":
                    selected = flat[should_steer.detach().reshape(-1)]
                else:
                    selected = flat

                selected_count = int(selected.shape[0])
                stats["histogram_weight_count"] = (
                    stats.get("histogram_weight_count", 0) + selected_count
                )
                if selected_count:
                    stats["histogram_alpha_sum"] = stats.get("histogram_alpha_sum", 0.0) + float(selected[:, 0].sum().item())
                    stats["histogram_beta_sum"] = stats.get("histogram_beta_sum", 0.0) + float(selected[:, 1].sum().item())
                    stats["histogram_gamma_sum"] = stats.get("histogram_gamma_sum", 0.0) + float(selected[:, 2].sum().item())
                    beta_idx = torch.floor(selected[:, 1] * bins).long()
                    gamma_idx = torch.floor(selected[:, 2] * bins).long()
                    valid = (
                        (beta_idx >= 0)
                        & (gamma_idx >= 0)
                        & (beta_idx < bins)
                        & (gamma_idx < bins)
                        & (beta_idx + gamma_idx < bins)
                    )
                    linear = beta_idx[valid] * bins + gamma_idx[valid]
                    counts = torch.bincount(linear, minlength=bins * bins)
                    stats["weight_histogram"] += counts.reshape(bins, bins).cpu().numpy()
        return (edited,) + rest if rest is not None else edited

    return hook


def main():
    parser = argparse.ArgumentParser(description="Evaluate S2S_R spherical steering")
    parser.add_argument("model_name")
    parser.add_argument("--checkpoint_path", "--operator_path", dest="checkpoint_path", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--weight_kappa", "--adaptive_kappa", dest="weight_kappa", type=float, default=8.0, help="Similarity attention sharpness for adaptive weights")
    parser.add_argument("--margin", type=float, default=-0.3, help="Only steer when max(beta, gamma) > alpha + margin")
    parser.add_argument("--method", choices=["S2S_R", "S2S_R_Static"], default="S2S_R")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--slerp_eps", type=float, default=1e-8)
    parser.add_argument("--small_angle_threshold", type=float, default=1e-4)
    parser.add_argument(
        "--residual_noise_sigma", type=float, default=0.0,
        help="Inference-only relative Gaussian noise on the predicted residual direction.",
    )
    parser.add_argument(
        "--residual_noise_seed", type=int, default=42,
        help="Seed for inference-only residual noise.",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument(
        "--weights_output_path",
        default=None,
        help="Optional JSON path for token-level alpha/beta/gamma histogram data.",
    )
    parser.add_argument(
        "--weights_bins",
        type=int,
        default=80,
        help="Number of beta/gamma bins per axis for --weights_output_path.",
    )
    parser.add_argument(
        "--weights_gate_mode",
        "--weights-gate-mode",
        choices=["all", "steered"],
        default="all",
        help=(
            "Tokens included in --weights_output_path: 'all' keeps every "
            "adaptive-weight token; 'steered' keeps only tokens passing the margin gate."
        ),
    )
    args = parser.parse_args()

    if args.weights_bins < 8:
        parser.error("--weights_bins must be at least 8")
    if not np.isfinite(args.residual_noise_sigma) or args.residual_noise_sigma < 0:
        parser.error("--residual_noise_sigma must be finite and non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data, avg, predictor = load_components(args.checkpoint_path, device)
    print(
        f"S2S_R | dim={avg.numel()} mlp_hidden={predictor.hidden_dim} "
        f"adaptive_kappa={args.weight_kappa} margin={args.margin}"
    )
    model_path = args.model_dir or HF_NAMES.get(args.model_name, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    layer = get_layer_name(args.model_name, args.layer)
    if args.output_path is None:
        args.output_path = f"results/{args.model_name.replace('/', '_')}_l{args.layer}_S2S_R_{mode}.json"
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "steered": 0, "weight_count": 0}
    if args.weights_output_path:
        stats["weight_histogram_bins"] = int(args.weights_bins)
        stats["weight_histogram"] = np.zeros(
            (args.weights_bins, args.weights_bins), dtype=np.int64
        )
        stats["histogram_weight_count"] = 0
        stats["histogram_alpha_sum"] = 0.0
        stats["histogram_beta_sum"] = 0.0
        stats["histogram_gamma_sum"] = 0.0
    hook = None if args.no_intervention else get_s2s_r_intervention(
        avg, predictor,
        weight_kappa=args.weight_kappa,
        margin=args.margin,
        eps=args.slerp_eps,
        small_angle_threshold=args.small_angle_threshold,
        stats=stats,
        weight_histogram_gate=args.weights_gate_mode,
        method=args.method,
        residual_noise_sigma=args.residual_noise_sigma,
        residual_noise_seed=args.residual_noise_seed,
    )
    progress = tqdm(range(len(results)), desc=mode)
    for i in progress:
        row = results.iloc[i]
        true_answers = split_multi_answer(row["Correct Answers"])
        false_answers = split_multi_answer(row["Incorrect Answers"])
        common = dict(
            model=model, tokenizer=tokenizer, question=row["Question"],
            base_hook_fn=hook, layer_name=layer, device=device,
            use_instruction=not args.no_instruction, model_name=args.model_name,
        )
        if args.few_shot:
            scores_true = calculate_scores_fewshot(choices=true_answers, preset=args.preset, **common)
            scores_false = calculate_scores_fewshot(choices=false_answers, preset=args.preset, **common)
        else:
            scores_true = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in true_answers]
            scores_false = [calculate_scores_zeroshot(choices=[a], **common)[0] for a in false_answers]
        MC_calcs(tag, results, i, scores_true, scores_false, true_answers, format_best(row["Best Answer"]))
        running = {m: float(results[f"{tag} {m}"].iloc[: i + 1].mean()) for m in ("MC1", "MC2", "MC3")}
        token_rate = 100.0 * stats["steered"] / max(stats["total"], 1)
        if stats["weight_count"]:
            mean_weights = tuple(stats[k] / stats["weight_count"] for k in ("alpha_sum", "beta_sum", "gamma_sum"))
            weight_text = f" W:{mean_weights[0]:.2f}/{mean_weights[1]:.2f}/{mean_weights[2]:.2f}"
        else:
            weight_text = ""
        progress.set_description(
            f"MC1:{running['MC1']:.3f} MC2:{running['MC2']:.3f} "
            f"MC3:{running['MC3']:.3f} TokenSteer:{token_rate:.1f}%{weight_text}"
        )
        if args.save_details:
            results.loc[i, f"{tag} scores_true"] = str(scores_true)
            results.loc[i, f"{tag} scores_false"] = str(scores_false)

    # Write a readable report: summary metrics first, followed by the
    # original per-question records under ``results``.  ``indent`` keeps the
    # summary and raw records visually separated by newlines.
    metric_values = {
        metric: float(results[f"{tag} {metric}"].mean())
        for metric in ("MC1", "MC2", "MC3")
    }
    token_rate = 100.0 * stats["steered"] / max(stats["total"], 1)
    metrics = {
        **metric_values,
        "token_steer_rate_percent": token_rate,
        "steered_tokens": int(stats["steered"]),
        "total_tokens": int(stats["total"]),
    }
    if stats["weight_count"]:
        metrics["mean_adaptive_weights"] = {
            "alpha": stats["alpha_sum"] / stats["weight_count"],
            "beta": stats["beta_sum"] / stats["weight_count"],
            "gamma": stats["gamma_sum"] / stats["weight_count"],
        }
    if args.weights_output_path:
        weights_path = Path(args.weights_output_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        histogram_count = int(stats.get("histogram_weight_count", 0))
        if histogram_count <= 0:
            raise RuntimeError(
                "No tokens were selected for the adaptive-weight histogram. "
                f"weights_gate_mode={args.weights_gate_mode!r}, margin={args.margin}."
            )
        weight_report = {
            "experiment": "fig7_adaptive_weight_density",
            "model_name": args.model_name,
            "layer": int(args.layer),
            "checkpoint_path": str(args.checkpoint_path),
            "eval_split": args.eval_split,
            "weight_kappa": float(args.weight_kappa),
            "margin": float(args.margin),
            "weights_gate_mode": args.weights_gate_mode,
            "bins": int(args.weights_bins),
            "coordinates": {"row": "beta", "column": "gamma", "alpha": "1-beta-gamma"},
            "token_count": histogram_count,
            "all_weight_token_count": int(stats["weight_count"]),
            "steered_token_count": int(stats["steered"]),
            "mean_adaptive_weights": {
                "alpha": stats["histogram_alpha_sum"] / histogram_count,
                "beta": stats["histogram_beta_sum"] / histogram_count,
                "gamma": stats["histogram_gamma_sum"] / histogram_count,
            },
            "histogram": stats["weight_histogram"].tolist(),
        }
        weights_path.write_text(
            json.dumps(weight_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metrics["weight_histogram_path"] = str(weights_path)
        metrics["weight_histogram_bins"] = int(args.weights_bins)
        metrics["weight_histogram_gate_mode"] = args.weights_gate_mode
        metrics["weight_histogram_token_count"] = histogram_count
    report = {
        "method": "S2S_R",
        "model_name": args.model_name,
        "layer": int(args.layer),
        "eval_split": args.eval_split,
        "checkpoint_path": str(args.checkpoint_path),
        "parameters": {
            "weight_kappa": args.weight_kappa,
            "margin": args.margin,
            "few_shot": bool(args.few_shot),
            "preset": args.preset,
            "residual_noise_sigma": float(args.residual_noise_sigma),
            "residual_noise_seed": int(args.residual_noise_seed),
        },
        "metrics": metrics,
        "results": json.loads(results.to_json(orient="records", force_ascii=False)),
    }
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved results to {args.output_path}")
    print(f"Token steer rate: {token_rate:.4f}% ({stats['steered']}/{stats['total']} tokens)")
    if stats["weight_count"]:
        print(
            "Adaptive weights mean (alpha,beta,gamma): "
            f"{stats['alpha_sum'] / stats['weight_count']:.6f}, "
            f"{stats['beta_sum'] / stats['weight_count']:.6f}, "
            f"{stats['gamma_sum'] / stats['weight_count']:.6f}"
        )


if __name__ == "__main__":
    main()
