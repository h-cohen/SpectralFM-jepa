"""Measurement-style spectral perturbations applied in raw units to globally normalized batches."""
from __future__ import annotations

import torch

KEYS = ("sigma_max", "offset_max", "slope_max", "scale_max", "shift_max", "prob")


def check_config(cfg: dict):
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        raise ValueError(f"training.robust missing {missing}")
    if not 0 <= cfg["prob"] <= 1 or cfg["shift_max"] not in (0, 1):
        raise ValueError("training.robust needs prob in [0, 1] and shift_max in {0, 1}")


def shift_one(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Per-row one-bin shift with edge padding. direction [B] in {-1, 0, 1}: +1 moves values left."""
    left = torch.cat([x[:, 1:], x[:, -1:]], 1)
    right = torch.cat([x[:, :1], x[:, :-1]], 1)
    d = direction[:, None]
    return torch.where(d > 0, left, torch.where(d < 0, right, x))


def perturb(z: torch.Tensor, cfg: dict, stats: dict, generator: torch.Generator) -> torch.Tensor:
    """z [B, L] globally normalized -> perturbed copy, same normalization.

    Each of noise, offset, slope, scale and shift is applied independently with probability `prob`.
    Noise SD and offset/slope sizes are relative to each spectrum's raw SD and range. All draws come
    from `generator` (CPU) so the stream is independent of masking and identical across arms.
    """
    B, L = z.shape
    mean, std = stats["mean"], stats["std"]
    x = z.float() * std + mean
    draw = lambda *shape: torch.rand(*shape, generator=generator).to(z.device)
    on = draw(5, B) < cfg["prob"]                                                  # [5, B]
    spread = x.amax(1, keepdim=True) - x.amin(1, keepdim=True)                     # [B, 1]
    sigma = draw(B, 1) * cfg["sigma_max"] * x.std(1, keepdim=True)
    noise = torch.randn(B, L, generator=generator).to(z.device) * sigma
    offset = (2 * draw(B, 1) - 1) * cfg["offset_max"] * spread
    ramp = torch.linspace(-.5, .5, L, device=z.device)[None]
    slope = (2 * draw(B, 1) - 1) * cfg["slope_max"] * spread * ramp
    scale = 1 + (2 * draw(B, 1) - 1) * cfg["scale_max"]
    direction = torch.where(draw(B) < .5, -1, 1) * cfg["shift_max"]
    x = torch.where(on[3, :, None], x * scale, x)
    x = x + on[0, :, None] * noise + on[1, :, None] * offset + on[2, :, None] * slope
    x = shift_one(x, torch.where(on[4], direction, 0))
    return ((x - mean) / std).to(z.dtype)
