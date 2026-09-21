"""L2S-LINEAR: standard additive steering ``h_new = h + alpha * v_hat``."""
from __future__ import annotations

import math
import torch


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


def get_l2s_linear_intervention(
    mlp,
    *,
    alpha: float = 1.0,
    stats: dict | None = None,
):
    """Create a TraceDict-compatible standard additive L2S hook.

    The MLP receives the raw hidden state exactly as in training and predicts
    ``v_hat``. Inference then applies ``h_new = h + alpha * v_hat`` directly.
    """
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
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
