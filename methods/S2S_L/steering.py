"""Core S2S_L steering primitives.

S2S_L reuses the direct-vector MLP trained by ``methods/S2S_R``. At
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
broadcast over the segment, matching the existing S2S_R evaluation semantics.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from methods.S2S_R.predictor import DirectVectorMLP, predict_residual
from methods.S2S_R.evaluate_S2S_R import adaptive_3way_weights
from methods.inference_residual_noise import make_noise_generator, perturb_unit_direction


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


def get_s2s_l_intervention(
    avg: torch.Tensor,
    predictor: DirectVectorMLP,
    *,
    strength: float = 1.0,
    slerp_t: float = 0.5,
    eps: float = 1e-8,
    small_angle_threshold: float = 1e-4,
    stats: dict | None = None,
    method: str = "S2S_L_Static",
    weight_kappa: float = 3.0,
    residual_noise_sigma: float = 0.0,
    residual_noise_seed: int = 42,
):
    """Build the S2S_L additive-steering hook.

    For each hook invocation, one ``v_hat`` is predicted from the last position
    of the intervention segment. The resulting S2S_L vector is broadcast to all
    positions in that segment, exactly as the existing S2S_R residual is.
    """
    if method not in {"S2S_L", "S2S_L_Static"}:
        raise ValueError(f"Unsupported S2S_L method: {method}")
    if not np.isfinite(strength):
        raise ValueError("strength must be finite")
    if not np.isfinite(slerp_t) or not 0.0 <= slerp_t <= 1.0:
        raise ValueError("slerp_t must be finite and in [0, 1]")
    if not np.isfinite(residual_noise_sigma) or residual_noise_sigma < 0:
        raise ValueError("residual_noise_sigma must be finite and non-negative")

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

        # Preserve S2S_R's sequence-level predictor semantics: use the last
        # position of the intervention segment to obtain one v_hat / residual.
        v_hat, residual = predict_residual(
            predictor,
            original[:, -1, :],
            avg,
            eps=eps,
        )
        v_hat = v_hat.float()
        residual = residual.float()
        if residual_noise_sigma > 0:
            if noise_generator is None:
                noise_generator = make_noise_generator(residual_noise_seed, v_hat.device)
            residual = perturb_unit_direction(
                v_hat - avg,
                sigma=residual_noise_sigma,
                generator=noise_generator,
                eps=eps,
            ).float()

        mean_batch = mean_direction.view(1, -1).expand_as(residual)
        if method == "S2S_L":
            # Use the same adaptive similarity graph as S2S_R, then reduce the
            # V/R weights to a 2-way spherical interpolation coefficient.
            x_last = norm(original[:, -1, :], eps)
            _, beta, gamma, _ = adaptive_3way_weights(
                x_last, mean_batch, residual, kappa=weight_kappa, temperature=1.0
            )
            dynamic_t = gamma / (beta + gamma).clamp_min(eps)
            t = dynamic_t
        else:
            t = slerp_t
        direction = slerp(
            mean_batch, residual, t, eps=eps,
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
