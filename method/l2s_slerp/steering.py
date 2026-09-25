"""L2S-SLERP: dynamic L2S prediction followed by gated spherical steering."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from method.inference_residual_noise import make_noise_generator, perturb_unit_direction


def _unpack_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:], tuple
    if isinstance(output, list):
        return output[0], output[1:], list
    return output, None, None


def _repack(hidden, rest, kind):
    if kind is tuple:
        return (hidden,) + rest
    if kind is list:
        return [hidden, *rest]
    return hidden


def _reconstruct_noisy_prediction(
    predicted: torch.Tensor,
    v_avg: torch.Tensor | None,
    *,
    sigma: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Perturb only ``predicted - v_avg`` while preserving its magnitude."""
    if sigma == 0.0:
        return predicted
    if v_avg is None:
        raise ValueError("v_avg is required when residual_noise_sigma is nonzero")
    avg = torch.as_tensor(v_avg, dtype=torch.float32, device=predicted.device)
    while avg.ndim < predicted.ndim:
        avg = avg.unsqueeze(0)
    if avg.shape[-1] != predicted.shape[-1]:
        raise ValueError("v_avg and predicted vectors must have the same hidden dimension")
    residual = predicted.float() - avg
    magnitude = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    direction = perturb_unit_direction(residual, sigma=sigma, generator=generator)
    return avg + magnitude * direction.float()


def l2s_slerp_transform(
    hidden: torch.Tensor,
    predicted_vectors: torch.Tensor,
    *,
    kappa: float,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the repository's current L2S spherical/vMF steering rule.

    ``hidden`` and ``predicted_vectors`` stay on their raw scale until this
    inference transform. Their directions are normalized for the spherical
    computation and the steered hidden state is rescaled to its original norm.
    The returned boolean mask marks positions where the vMF gate fired.
    """
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(beta) or beta >= 1.0:
        raise ValueError("beta must be finite and smaller than 1")
    if hidden.shape != predicted_vectors.shape:
        raise ValueError("hidden and predicted_vectors must have the same shape")

    raw_hidden = hidden.float()
    raw_pred = predicted_vectors.float()
    hidden_norm = torch.linalg.vector_norm(raw_hidden, dim=-1, keepdim=True).clamp_min(eps)
    pred_norm = torch.linalg.vector_norm(raw_pred, dim=-1, keepdim=True)
    x_hat = raw_hidden / hidden_norm
    mu_t = raw_pred / pred_norm.clamp_min(eps)

    cos_t = (x_hat * mu_t).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cos_h = -cos_t
    logits = torch.cat([kappa * cos_t, kappa * cos_h], dim=-1)
    probs = F.softmax(logits, dim=-1)
    delta = probs[..., 1:2] - probs[..., 0:1]

    theta = torch.acos(cos_t)
    sin_theta = torch.sin(theta)
    triggered = (delta > beta) & (theta >= small_angle_threshold) & (pred_norm > eps)
    t = alpha * (delta - beta) / (1.0 - beta)
    t = torch.clamp(t, 0.0, 1.0)
    orthogonal = (x_hat - cos_t * mu_t) / sin_theta.clamp_min(eps)
    theta_new = (1.0 - t) * theta
    steered_hat = torch.cos(theta_new) * mu_t + torch.sin(theta_new) * orthogonal
    steered = steered_hat * hidden_norm
    edited = torch.where(triggered, steered, raw_hidden)
    return edited.to(hidden.dtype), triggered.squeeze(-1)


def get_l2s_slerp_intervention(
    mlp,
    *,
    kappa: float = 20.0,
    alpha: float = 0.15,
    beta: float = 0.1,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
    stats: dict | None = None,
    v_avg: torch.Tensor | None = None,
    residual_noise_sigma: float = 0.0,
    residual_noise_seed: int = 42,
):
    """Create a TraceDict-compatible L2S-SLERP hook."""
    if not np.isfinite(residual_noise_sigma) or residual_noise_sigma < 0:
        raise ValueError("residual_noise_sigma must be finite and non-negative")
    mlp.eval()
    mlp_device = next(mlp.parameters()).device
    noise_generator = None

    def hook(output, layer_name=None, start_idx: int | None = None):
        nonlocal noise_generator
        del layer_name
        hidden, rest, output_kind = _unpack_output(output)
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output
        seq_len = hidden.shape[1]
        start = max(0, min(int(start_idx if start_idx is not None else seq_len - 1), seq_len - 1))
        segment = hidden[:, start:, :]
        raw_input = segment.float().reshape(-1, segment.shape[-1])
        predicted = mlp(raw_input.to(mlp_device)).to(hidden.device).reshape_as(segment)
        if residual_noise_sigma > 0 and noise_generator is None:
            noise_generator = make_noise_generator(residual_noise_seed, hidden.device)
        predicted = _reconstruct_noisy_prediction(
            predicted,
            v_avg,
            sigma=residual_noise_sigma,
            generator=noise_generator,
        )
        edited_segment, triggered = l2s_slerp_transform(
            segment,
            predicted,
            kappa=kappa,
            alpha=alpha,
            beta=beta,
            eps=eps,
            small_angle_threshold=small_angle_threshold,
        )
        edited = hidden.clone()
        edited[:, start:, :] = edited_segment

        if stats is not None:
            total = int(triggered.numel())
            steered = int(triggered.sum().item())
            stats["total"] = stats.get("total", 0) + total
            stats["steered"] = stats.get("steered", 0) + steered
            stats["steer_rate"] = stats["steered"] / max(stats["total"], 1)
        return _repack(edited, rest, output_kind)

    return hook
