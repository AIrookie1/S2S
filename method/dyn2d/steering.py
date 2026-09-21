"""Core Dyn2D steering primitives.

Dyn2D reuses the direct-vector MLP trained by ``method/dyn3d_sga``. At
inference time it forms two unit directions:

    V = normalize(v_avg)
    R = normalize(v_hat - v_avg)

and interpolates them on the unit sphere:

    D = SLERP(V, R, slerp_t)

The direction is then restored to the original predicted-vector magnitude
``||v_hat||_2`` and applied with standard additive activation steering:

    D_scaled = D * ||v_hat||_2
    h_new = h + strength * D_scaled

One prediction is made from the last position in the intervention segment and
broadcast over the segment, matching the existing Dyn3D evaluation semantics.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from method.dyn3d_sga.dyn_mlp import DirectVectorMLP, predict_residual


def norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the final dimension in float32."""
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def _broadcast_t(t, reference: torch.Tensor) -> torch.Tensor:
    t_tensor = torch.as_tensor(t, dtype=reference.dtype, device=reference.device)
    while t_tensor.ndim < reference.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
    return t_tensor


def _deterministic_orthogonal(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return a deterministic unit vector orthogonal to each unit vector in x."""
    axis_idx = x.abs().argmin(dim=-1, keepdim=True)
    basis = torch.zeros_like(x).scatter_(-1, axis_idx, 1.0)
    orthogonal = basis - (basis * x).sum(dim=-1, keepdim=True) * x
    return F.normalize(orthogonal, p=2, dim=-1, eps=eps)


def slerp(
    x: torch.Tensor,
    y: torch.Tensor,
    t: float | torch.Tensor,
    *,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
) -> torch.Tensor:
    """Numerically stable spherical linear interpolation of two directions.

    ``t`` must lie in [0, 1]. Endpoints are exact after normalization. Small
    angles use normalized linear interpolation. Near-antipodal vectors use a
    deterministic great-circle fallback so the midpoint never collapses to a
    zero vector.
    """
    if not np.isfinite(small_angle_threshold) or small_angle_threshold <= 0:
        raise ValueError("small_angle_threshold must be finite and positive")

    x_unit = norm(x, eps)
    y_unit = norm(y, eps)
    t_tensor = _broadcast_t(t, (x_unit * y_unit).sum(dim=-1, keepdim=True))
    if not bool(torch.isfinite(t_tensor).all()):
        raise ValueError("slerp_t must be finite")
    if bool(((t_tensor < 0) | (t_tensor > 1)).any()):
        raise ValueError("slerp_t must be in [0, 1]")

    dot = (x_unit * y_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)

    # Stable great-circle form: y = cos(theta)x + sin(theta)u.
    orthogonal = y_unit - dot * x_unit
    orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    orthogonal_unit = orthogonal / orthogonal_norm.clamp_min(eps)
    spherical = (
        torch.cos(t_tensor * theta) * x_unit
        + torch.sin(t_tensor * theta) * orthogonal_unit
    )

    lerped = norm((1.0 - t_tensor) * x_unit + t_tensor * y_unit, eps)
    near_parallel = theta <= small_angle_threshold

    anti_distance = math.pi - theta
    near_antipodal = anti_distance <= small_angle_threshold
    anti_orthogonal = _deterministic_orthogonal(x_unit, eps)
    antipodal = (
        torch.cos(math.pi * t_tensor) * x_unit
        + torch.sin(math.pi * t_tensor) * anti_orthogonal
    )

    result = torch.where(near_parallel, lerped, spherical)
    result = torch.where(near_antipodal, antipodal, result)
    result = norm(result, eps)

    # Guarantee exact normalized endpoints, including the antipodal fallback.
    result = torch.where(t_tensor == 0, x_unit, result)
    result = torch.where(t_tensor == 1, y_unit, result)
    return result


def get_2d_dynamic_intervention(
    avg: torch.Tensor,
    predictor: DirectVectorMLP,
    *,
    strength: float = 1.0,
    slerp_t: float = 0.5,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
    stats: dict | None = None,
):
    """Build the Dyn2D additive-steering hook.

    For each hook invocation, one ``v_hat`` is predicted from the last position
    of the intervention segment. The resulting Dyn2D vector is broadcast to all
    positions in that segment, exactly as the existing Dyn3D residual is.
    """
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    if not np.isfinite(slerp_t) or not 0.0 <= slerp_t <= 1.0:
        raise ValueError("slerp_t must be finite and in [0, 1]")

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
        original = hidden[:, start:, :].float()

        # Preserve Dyn3D's sequence-level predictor semantics: use the last
        # position of the intervention segment to obtain one v_hat / residual.
        v_hat, residual = predict_residual(
            predictor,
            original[:, -1, :],
            avg,
            eps=eps,
        )
        v_hat = v_hat.float()
        residual = residual.float()

        mean_batch = mean_direction.view(1, -1).expand_as(residual)
        direction = slerp(
            mean_batch,
            residual,
            slerp_t,
            eps=eps,
            small_angle_threshold=small_angle_threshold,
        )
        predicted_norm = torch.linalg.vector_norm(v_hat, dim=-1, keepdim=True)
        scaled_direction = direction * predicted_norm
        steering = float(strength) * scaled_direction[:, None, :]

        edited = hidden.clone()
        edited[:, start:, :] = (original + steering).to(hidden.dtype)

        if stats is not None:
            n_tokens = int(original.shape[0] * original.shape[1])
            stats["total"] = stats.get("total", 0) + n_tokens
            if strength != 0:
                stats["steered"] = stats.get("steered", 0) + n_tokens
            else:
                stats.setdefault("steered", 0)
            stats["prediction_count"] = stats.get("prediction_count", 0) + int(v_hat.shape[0])
            stats["predicted_norm_sum"] = stats.get("predicted_norm_sum", 0.0) + float(predicted_norm.sum().item())
            stats["scaled_norm_sum"] = stats.get("scaled_norm_sum", 0.0) + float((predicted_norm * abs(float(strength))).sum().item())

        return (edited,) + rest if rest is not None else edited

    return hook
