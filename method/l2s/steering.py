"""Dynamic L2S prediction followed by Spherical-Steering interpolation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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


def spherical_l2s_transform(
    hidden: torch.Tensor,
    predicted_vectors: torch.Tensor,
    *,
    kappa: float,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized form of ``spherical_geometric_logic`` from Spherical-Steering.

    ``hidden`` and ``predicted_vectors`` are raw-scale vectors.  Both are
    normalized only for the spherical computation; the output is rescaled to
    each hidden state's original norm.  The returned boolean mask identifies
    positions where the same vMF gate as the reference implementation fired.
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


def get_l2s_spherical_intervention(
    mlp,
    *,
    kappa: float = 20.0,
    alpha: float = 0.15,
    beta: float = 0.1,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
    stats: dict | None = None,
):
    """Create a TraceDict-compatible dynamic L2S spherical hook.

    The MLP receives the raw hidden state (matching raw-scale training).  Only
    the hidden/predicted-vector copies used by spherical steering are
    normalized, exactly as requested for L2S inference.
    """
    mlp.eval()
    mlp_device = next(mlp.parameters()).device

    def hook(output, layer_name=None, start_idx: int | None = None):
        del layer_name
        hidden, rest, output_kind = _unpack_output(output)
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output
        seq_len = hidden.shape[1]
        start = max(0, min(int(start_idx if start_idx is not None else seq_len - 1), seq_len - 1))
        segment = hidden[:, start:, :]
        raw_input = segment.float().reshape(-1, segment.shape[-1])
        predicted = mlp(raw_input.to(mlp_device)).to(hidden.device).reshape_as(segment)
        edited_segment, triggered = spherical_l2s_transform(
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
