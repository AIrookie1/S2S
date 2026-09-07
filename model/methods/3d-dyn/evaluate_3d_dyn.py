"""Three-vector spherical dynSVD steering with token-level live statistics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.methods.dynSVD.dynSVD import normalize_basis_combination
from model.methods.dynSVD.evaluate_dynSVD import load_components
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

    ``x``, ``v`` and ``r`` must already be unit vectors.  Keeping that
    contract explicit avoids repeatedly normalizing the same vectors in the
    hot inference hook: the caller normalizes the raw hidden state once,
    ``v`` is pre-normalized once per evaluation, and ``r`` is already
    normalized by ``normalize_basis_combination``.
    """
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("weight kappa must be finite and positive")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("weight temperature must be finite and positive")

    s_xv = (x * v).sum(dim=-1, keepdim=True)
    s_xr = (x * r).sum(dim=-1, keepdim=True)
    s_vr = (v * r).sum(dim=-1, keepdim=True)

    attn_x = torch.softmax(kappa * torch.cat([s_xv, s_xr], dim=-1), dim=-1)
    a_xv, a_xr = attn_x[..., 0:1], attn_x[..., 1:2]
    attn_v = torch.softmax(kappa * torch.cat([s_xv, s_vr], dim=-1), dim=-1)
    a_vx, a_vr = attn_v[..., 0:1], attn_v[..., 1:2]
    attn_r = torch.softmax(kappa * torch.cat([s_xr, s_vr], dim=-1), dim=-1)
    a_rx, a_rv = attn_r[..., 0:1], attn_r[..., 1:2]

    centrality = torch.cat([(a_vx + a_rx), (a_xv + a_rv), (a_xr + a_vr)], dim=-1)
    # print((a_vx + a_rx), (a_xv + a_rv), (a_xr + a_vr))
    weights = torch.softmax(centrality / temperature, dim=-1)
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
    linear = norm((1 - t_tensor) * x + t_tensor * y, eps)
    return torch.where(theta.abs() < small_angle_threshold, linear, result)


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
    return norm(alpha * first + beta * second + gamma * third, eps)


def validate_weights(alpha, beta, gamma):
    if not all(np.isfinite(x) and x >= 0 for x in (alpha, beta, gamma)):
        raise ValueError("alpha, beta, gamma must be finite and non-negative")
    if not np.isclose(alpha + beta + gamma, 1.0, atol=1e-6):
        raise ValueError(
            f"alpha + beta + gamma must equal 1 (got {alpha + beta + gamma:.8f})"
        )


def get_3d_dynamic_intervention(
    avg,
    basis,
    router,
    scale_net,
    weight_kappa=5.0,
    weight_temperature=0.5,
    hidden_cosine_threshold=1.0,
    eps=1e-8,
    small_angle_threshold=1e-4,
    stats=None,
):
    """Build the 3d rotation hook and count every considered token."""
    del scale_net  # 3d inference uses the learned router direction.
    if not np.isfinite(hidden_cosine_threshold):
        raise ValueError("hidden_cosine_threshold must be finite")

    mean_direction = norm(avg, eps)

    def hook(output, layer_name=None, start_idx=None):
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
        z = hidden[:, start, :].float()
        coeff = torch.softmax(router.logits(z), dim=-1)
        residual = normalize_basis_combination(coeff, basis).float()
        original = hidden[:, start:, :].float()
        original_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True).clamp_min(eps)
        a = norm(original, eps)
        b = mean_direction.view(1, 1, -1).expand_as(a)
        c = residual.view(residual.shape[0], 1, -1).expand_as(a)
        alpha, beta, gamma, weight_info = adaptive_3way_weights(
            a, b, c, kappa=weight_kappa, temperature=weight_temperature
        )
        rotated_direction = two_slerp_sym(
            a, b, c, alpha, beta, gamma, eps, small_angle_threshold
        )
        candidate = rotated_direction * original_norm
        cosine = F.cosine_similarity(candidate, original, dim=-1)
        # print(f"Cosine similarity: {cosine.mean().item()}")
        gate = cosine < hidden_cosine_threshold
        edited = hidden.clone()
        edited[:, start:, :] = torch.where(gate[..., None], candidate, original).to(hidden.dtype)

        if stats is not None:
            token_count = int(gate.numel())
            steered_count = int(gate.sum().item())
            stats["total"] = stats.get("total", 0) + token_count
            stats["steered"] = stats.get("steered", 0) + steered_count
            stats["last_batch_rate"] = steered_count / max(token_count, 1)
            stats["weight_count"] = stats.get("weight_count", 0) + token_count
            stats["alpha_sum"] = stats.get("alpha_sum", 0.0) + float(alpha.sum().item())
            stats["beta_sum"] = stats.get("beta_sum", 0.0) + float(beta.sum().item())
            stats["gamma_sum"] = stats.get("gamma_sum", 0.0) + float(gamma.sum().item())
            stats["last_weight_mean"] = weight_info["weights"].mean(dim=tuple(range(weight_info["weights"].ndim - 1))).detach().cpu().tolist()
        return (edited,) + rest if rest is not None else edited

    return hook


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3d-dyn spherical steering")
    parser.add_argument("model_name")
    parser.add_argument("--checkpoint_path", "--operator_path", dest="checkpoint_path", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--csv_path", default="./TruthfulQA/data/v1/TruthfulQA.csv")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--weight_kappa", "--adaptive_kappa", dest="weight_kappa", type=float, default=5.0, help="Similarity attention sharpness for adaptive weights")
    parser.add_argument("--weight_temperature", "--adaptive_temperature", dest="weight_temperature", type=float, default=0.5, help="Softmax temperature for adaptive weights")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--preset", default="qa")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--no_intervention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--hidden_cosine_threshold", type=float, default=0.99)
    parser.add_argument("--slerp_eps", type=float, default=1e-8)
    parser.add_argument("--small_angle_threshold", type=float, default=1e-4)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--save_details", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    data, avg, basis, router, scale_net = load_components(args.checkpoint_path, device)
    print(
        f"3d-dyn | dim={basis.shape[1]} rank={basis.shape[0]} "
        f"adaptive_kappa={args.weight_kappa} temperature={args.weight_temperature}"
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
        args.output_path = f"results/{args.model_name.replace('/', '_')}_l{args.layer}_3d_dyn_{mode}.json"
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "steered": 0, "weight_count": 0}
    hook = None if args.no_intervention else get_3d_dynamic_intervention(
        avg, basis, router, scale_net,
        weight_kappa=args.weight_kappa,
        weight_temperature=args.weight_temperature,
        hidden_cosine_threshold=args.hidden_cosine_threshold,
        eps=args.slerp_eps,
        small_angle_threshold=args.small_angle_threshold,
        stats=stats,
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

    results.to_json(args.output_path, orient="records", force_ascii=False)
    token_rate = 100.0 * stats["steered"] / max(stats["total"], 1)
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
