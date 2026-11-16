"""L2S-LINEAR: standard additive steering ``h_new = h + alpha * v_hat``."""
from __future__ import annotations

import math
import torch

from methods.inference_residual_noise import make_noise_generator, perturb_unit_direction


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


def l2s_linear_transform(
    hidden: torch.Tensor,
    predicted_vectors: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply standard raw-scale additive steering with no normalization/gate."""
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if hidden.shape != predicted_vectors.shape:
        raise ValueError("hidden and predicted_vectors must have the same shape")
    edited = hidden.float() + float(alpha) * predicted_vectors.float()
    return edited.to(hidden.dtype)


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


def get_l2s_linear_intervention(
    mlp,
    *,
    alpha: float = 1.0,
    stats: dict | None = None,
    v_avg: torch.Tensor | None = None,
    residual_noise_sigma: float = 0.0,
    residual_noise_seed: int = 42,
):
    """Create a TraceDict-compatible standard additive L2S hook.

    The MLP receives the raw hidden state exactly as in training and predicts
    ``v_hat``. Inference then applies ``h_new = h + alpha * v_hat`` directly.
    """
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not math.isfinite(residual_noise_sigma) or residual_noise_sigma < 0:
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
        edited_segment = l2s_linear_transform(segment, predicted, alpha=alpha)
        edited = hidden.clone()
        edited[:, start:, :] = edited_segment

        if stats is not None:
            total = int(segment.shape[0] * segment.shape[1])
            stats["total"] = stats.get("total", 0) + total
            stats["steered"] = stats.get("steered", 0) + (total if alpha != 0.0 else 0)
            stats["steer_rate"] = stats["steered"] / max(stats["total"], 1)
        return _repack(edited, rest, output_kind)

    return hook
