"""VISReg (Wu, Balestriero, Levine 2026, arXiv 2606.02572): variance-invariance-sketching regularizer.

For z [..., N, D] (leading dims are independent tests, like lightly's SIGReg) it returns, averaged over the
leading dims, a center term (mean^2), a scale term ((1 - std)^2) and a shape term: the sorted random 1-D
projections of the std-normalized batch (stop-grad on std) against standard-normal quantiles.
One set of unit-norm directions is drawn per call from torch's global RNG on z's device.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VISReg(nn.Module):
    def __init__(self, num_slices: int = 4096, eps: float = 1e-4):
        super().__init__()
        self.num_slices = num_slices
        self.eps = eps

    def forward(self, z: torch.Tensor) -> dict:
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            n, d = z.shape[-2], z.shape[-1]
            mu = z.mean(dim=-2, keepdim=True)
            center = mu.pow(2).mean(dim=(-2, -1))
            zc = z - mu
            std = zc.std(dim=-2, unbiased=False, keepdim=True)
            scale = (1.0 - std).pow(2).mean(dim=(-2, -1))
            znorm = zc / (std.detach() + self.eps)
            w = torch.randn(d, self.num_slices, device=z.device)
            w = w / w.norm(dim=0, keepdim=True)
            proj = torch.sort(znorm @ w, dim=-2).values                     # [..., N, K]
            u = torch.arange(1, n + 1, device=z.device, dtype=torch.float32) / (n + 1)
            target = torch.special.ndtri(u).unsqueeze(-1)                   # [N, 1]
            shape = (proj - target).pow(2).mean(dim=(-2, -1))
            center, scale, shape = center.mean(), scale.mean(), shape.mean()
        return {"reg": center + scale + shape, "center": center, "scale": scale, "shape": shape}
