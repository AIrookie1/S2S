"""Deterministic inference-time noise for residual vector directions."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def make_noise_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a reproducible generator on the requested device."""
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def perturb_unit_direction(
    vector: torch.Tensor,
    *,
    sigma: float,
    generator: torch.Generator | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Add relative Gaussian direction noise and return a unit vector.

    The perturbation is ``normalize(v + sigma * ||v|| * normalize(epsilon))``.
    A zero sigma is deterministic and does not consume the RNG.
    """
    if not math.isfinite(float(sigma)) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    raw = vector.float()
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    direction = F.normalize(raw, p=2, dim=-1, eps=eps)
    if sigma == 0.0:
        return direction.to(vector.dtype)
    if generator is None:
        generator = make_noise_generator(0, raw.device)
    noise = torch.randn(
        raw.shape,
        generator=generator,
        device=raw.device,
        dtype=raw.dtype,
    )
    noise = F.normalize(noise, p=2, dim=-1, eps=eps)
    return F.normalize(raw + float(sigma) * norm * noise, p=2, dim=-1, eps=eps).to(vector.dtype)
